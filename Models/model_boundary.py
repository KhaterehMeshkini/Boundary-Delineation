import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34
from timm.models.layers import trunc_normal_
import torchvision
#from torch_geometric.nn import GCNConv

class Config:
   cnn_backbone ='resnet34'
   resnet_pretrained = True

config = Config()   

class Res_AttentionBlock3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Res_AttentionBlock3D, self).__init__()

        self.F_l = F_l
        self.F_g = F_g
        self.F_int = F_int
        self.W_x = nn.Sequential(
            nn.Conv3d(self.F_l, self.F_int, kernel_size=(1, 1, 1), stride=(2,2,2), padding=(0,0,0), bias=True),
            nn.BatchNorm3d(self.F_int)
        )

        self.W_g = nn.Sequential(
            nn.Conv3d(self.F_g, self.F_int, kernel_size=(1, 1, 1), stride=1, padding=(0,0,0), bias=True),
            nn.BatchNorm3d(self.F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(self.F_int, 1, kernel_size=(1, 1, 1), stride=1, padding=(0,0,0), bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)


    def forward(self, g, x):
            # Determine the temporal dimension from input tensors
        temporal_dim = g.shape[2]
        
        if temporal_dim > 1:
            kernel = 3 
            
        else:
            kernel = 1
        
        padding_size = kernel//2 

        #Determine device from input tensors
        device = x.device

        # Forward pass with the dynamically defined convolutions
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi_out = self.relu(g1 + x1)
        psi_out = self.psi(psi_out)

        #Upsample psi_out by a factor of 2 along spatial dimensions
        psi_out = F.interpolate(psi_out, scale_factor=(2,2,2), mode='trilinear', align_corners=False)
        
        return x * psi_out  # Applying attention to x instead of g   
    
     

class ConvWindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super(ConvWindowAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        self.qkv_conv = nn.Conv1d(dim, dim * 3, kernel_size=1)
        self.out_conv = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        B, N, C = x.shape  
        

        # Reshape input to (B, N, C) for compatibility with Conv1d
        #x = x.view(B, C, N).permute(0, 2, 1)  # Shape: (B, N, C)

        # Project to QKV space and reshape
        qkv = self.qkv_conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)

        # Separate q, k, v
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Scaled dot-product attention
        attention_scores = (q @ k.transpose(-2, -1)) * self.scale
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_output = (attention_scores @ v)

        # Reshape output back to (B, C, D, H, W)
        attention_output = attention_output.transpose(1, 2).reshape(B, N, C)
        x = self.out_conv(attention_output.permute(0, 2, 1)).permute(0, 2, 1)
        
        return x


class ConvWindowTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super(ConvWindowTransformerBlock, self).__init__()
        self.attention_layer = ConvWindowAttention(dim, num_heads, window_size)
        self.conv_layer = nn.Sequential(
            nn.Conv1d(dim, dim * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim * 4, dim, kernel_size=3, padding=1)
        )

    def forward(self, x):
        attention_output = self.attention_layer(x)
        x = x + attention_output

        conv_output = self.conv_layer(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + conv_output
        return x


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim):
        super(CrossAttentionFusion, self).__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.fusion_conv = nn.Conv3d(2 * dim, dim, kernel_size=1)


    def forward(self, transformer_features, cnn_features):
        # Reshape features
        B, C, D, H, W = cnn_features.shape
        cnn_features = cnn_features.view(B, C, D * H * W).transpose(1, 2)  # Shape: (B, N, C)
        transformer_features = transformer_features.view(B, C, D * H * W).transpose(1, 2)  # Shape: (B, N, C)
        
        # Project to Q, K, V
        key = self.key_proj(cnn_features)  # CNN output as Key
        value = self.value_proj(cnn_features)  # CNN output as Value
        query = self.query_proj(transformer_features)  # Transformer output as Query


        # Compute attention scores
        attention_scores = torch.bmm(query, key.transpose(1, 2)) / (C ** 0.5)
        attention_weights = self.softmax(attention_scores)  # Shape: (B, N, N)

        # Weighted sum of values (attention mechanism)
        attended_features = torch.bmm(attention_weights, value)

        # Reshape back to 3D form and add with transformer features for final fusion
        attended_features = attended_features.transpose(1, 2).view(B, C, D, H, W)
        transformer_features = transformer_features.transpose(1, 2).view(B, C, D, H, W)
        # Modify the final fusion step
        concatenated_features = torch.cat((transformer_features, attended_features), dim=1)
        fused_features = self.fusion_conv(concatenated_features)  # Additional convolution after concatenation

        #fused_features = transformer_features + attended_features  # Fusion by addition
        return fused_features
    
    
# Modify your encoder to use the CrossAttentionFusion module
class AdaptiveSwinTransformerEncoder(nn.Module):
    def __init__(self, in_channels, base_window_size=7, num_heads=[4, 8, 16]):
        super(AdaptiveSwinTransformerEncoder, self).__init__()

        
        # Load pre-trained ResNet34 model
        resnet = eval(f"torchvision.models.{config.cnn_backbone}(pretrained ={config.resnet_pretrained})")
        resnet_layers = list(resnet.children())
                # List of supported layer types for conversion
        supported_layers = (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d)

        self.resnet_layers = nn.ModuleList()

        for layer in resnet_layers:
            if isinstance(layer, supported_layers):
                if isinstance(layer, nn.Conv2d):
                    # Convert Conv2D to Conv3D
                    conv3d = nn.Conv3d(
                        in_channels=in_channels,
                        out_channels=layer.out_channels,
                        kernel_size=(1, layer.kernel_size[0], layer.kernel_size[1]),
                        stride=(1, layer.stride[0], layer.stride[1]),
                        padding=(0, layer.padding[0], layer.padding[1]),
                        bias=(layer.bias is not None)
                    )
                    # Adapt weights
                    original_weights = layer.weight.data  # Shape: (out_channels, 3, H, W)
                    
                    # Step 1: Expand RGB weights to 8 spectral bands
                    expanded_weights = original_weights.repeat(1, 4, 1, 1)[:, :10, :, :]  # Shape: (out_channels, 10, H, W)

                    # Step 2: Add depth axis (1, 1, H, W)
                    expanded_weights = expanded_weights.unsqueeze(2)  # Shape: (out_channels, 10, 1, H, W)

                    conv3d.weight.data = expanded_weights
                    if layer.bias is not None:
                        conv3d.bias.data = layer.bias.data
                    self.resnet_layers.append(conv3d)
                elif isinstance(layer, nn.BatchNorm2d):
                    # Convert BatchNorm2D to BatchNorm3D
                    bn3d = nn.BatchNorm3d(
                        num_features=layer.num_features,
                        eps=layer.eps,
                        momentum=layer.momentum,
                        affine=layer.affine,
                        track_running_stats=layer.track_running_stats
                    )
                    bn3d.weight.data = layer.weight.data
                    bn3d.bias.data = layer.bias.data
                    bn3d.running_mean = layer.running_mean
                    bn3d.running_var = layer.running_var
                    self.resnet_layers.append(bn3d)
                elif isinstance(layer, nn.ReLU):
                    self.resnet_layers.append(nn.ReLU(inplace=True))
                elif isinstance(layer, nn.MaxPool2d):
                    # Convert MaxPool2D to MaxPool3D
                    pool3d = nn.MaxPool3d(
                        kernel_size=(1, layer.kernel_size, layer.kernel_size),
                        stride=(1, layer.stride, layer.stride),
                        padding=(0, layer.padding, layer.padding)
                    )
                    self.resnet_layers.append(pool3d)
            else:
                print(f"Skipping unsupported layer: {type(layer)}")

        
        # Define CNN and Transformer layers
        self.conv_block1 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((8, 64, 64))
        )


        #swin = swin_tiny_patch4_window7_224(pretrained=True)
        self.transformer_block1 = ConvWindowTransformerBlock(dim=128, num_heads=num_heads[0], window_size=base_window_size)
        #swin_weights = swin.state_dict()
        #self.transformer_block1.load_state_dict({k: swin_weights[k] for k in self.transformer_block1.state_dict().keys()}, strict=False)

        self.cross_attention1 = CrossAttentionFusion(dim=128)

        self.conv_block2 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.transformer_block2 = ConvWindowTransformerBlock(dim=256, num_heads=num_heads[1], window_size=base_window_size)
        self.cross_attention2 = CrossAttentionFusion(dim=256)

        self.conv_block3 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.transformer_block3 = ConvWindowTransformerBlock(dim=512, num_heads=num_heads[2], window_size=base_window_size)
        self.cross_attention3 = CrossAttentionFusion(dim=512)

   

    def forward(self, x):

        for i in range(3):
            x= self.resnet_layers[i](x)
        # Level 1 features
        conv_x1 = self.conv_block1(x)
        B, C, D, H, W = conv_x1.shape
        x1_flattened = conv_x1.view(B, C, D * H * W).transpose(1, 2)
        trans_x1 = self.transformer_block1(x1_flattened).transpose(1, 2).view(B, C, D, H, W)
        x1 = trans_x1

        # Level 2 features
        conv_x2 = self.conv_block2(x1)
        B, C, D, H, W = conv_x2.shape
        x2_flattened = conv_x2.view(B, C, D * H * W).transpose(1, 2)
        trans_x2 = self.transformer_block2(x2_flattened).transpose(1, 2).view(B, C, D, H, W)
        x2 = trans_x2

        # Level 3 features
        conv_x3 = self.conv_block3(x2)
        B, C, D, H, W = conv_x3.shape
        x3_flattened = conv_x3.view(B, C, D * H * W).transpose(1, 2)
        trans_x3 = self.transformer_block3(x3_flattened).transpose(1, 2).view(B, C, D, H, W)
        x3 = trans_x3

        return x1, x2, x3 
        

class DynamicAttentionUNet3D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DynamicAttentionUNet3D, self).__init__()

        # Encoder with Adaptive Swin Transformer
        self.adaptive_swin_encoder = AdaptiveSwinTransformerEncoder(in_channels)

        # Intermediate Convolutional Layer
        self.intermediate_conv = nn.Conv3d(512, 1024, kernel_size=3, padding=1)
        self.intermediate_activation = nn.ReLU(inplace=True)
        self.intermediate_maxpooling = nn.MaxPool3d(kernel_size=2, stride=2)


        # Decoder (Remains unchanged)
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose3d(1024, 512, kernel_size=(2,3,3), stride=3),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(256)
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=(2,3,3), stride=3),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(64)
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=(2,3,3), stride=3),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(16)
        )


        
        #input 16
        self.final_layers = nn.Sequential(
            # Collapse temporal dimension from 8 → 1 using a 3D conv
            nn.Conv3d(16, 32, kernel_size=(8, 1, 1)),  # Output: (B, 32, 1, 64, 64)
            nn.ReLU(inplace=True),

            # Squeeze the temporal dimension
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # Treat (B, 32, 64, 64) as 2D
            nn.ReLU(inplace=True),

            # Upsample 64x64 → 128x128
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),  # 128x128
            nn.ReLU(inplace=True),

            # Upsample 128x128 → 256x256
            nn.ConvTranspose2d(16, 2, kernel_size=2, stride=2)  # Output: (B, 2, 256, 256)
        )


        # Attention Blocks (Unchanged)
        self.attention3 = Res_AttentionBlock3D(F_g=1024, F_l=512, F_int=512)
        self.attention2 = Res_AttentionBlock3D(F_g=512, F_l=256, F_int=256)
        self.attention1 = Res_AttentionBlock3D(F_g=256, F_l=128, F_int=128)

        # Final 1x1 convolution for binary classification
        self.temporal_pool1 = nn.Conv3d(64, 64, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        self.temporal_pool2 = nn.Conv3d(64, 64, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        self.spatial_upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2),  # 64 -> 128
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, num_classes, kernel_size=2, stride=2),  # 128 -> 256
        )



        
    def forward(self, data):
        # Encoder
        encoder_outputs = self.adaptive_swin_encoder(data)
        
        x1, x2, x3 = encoder_outputs

        # Intermediate Convolutional Layer
        x = self.intermediate_conv(x3)
        x = self.intermediate_activation(x)
        x = self.intermediate_maxpooling(x)

        # Decoder with Attention Gates
        xd = self.decoder3[0:1](x)
        #x3_att = self.attention3(g=x, x=x3)
        #x = torch.cat([xd, x3_att], dim=1)
        x = self.decoder3[2:](xd)

        xd = self.decoder2[0](x)  
        #x2_att = self.attention2(g=x, x=x2)
        #x = torch.cat([xd, x2_att], dim=1)
        x = self.decoder2[2:](xd)

        xd = self.decoder1[0](x) 
        # x1_att = self.attention1(g=x, x=x1)
        # x = torch.cat([xd, x1_att], dim=1)
        x = self.decoder1[2:](xd)

        # Final 1x1 convolution for binary classification
        # out1 = self.temporal_pool1(x)
        # out2 = self.temporal_pool1(out1)
        # out3 = F.adaptive_avg_pool3d(out2, (1, out2.size(-2), out2.size(-1))) 

        # out3 = out3.squeeze(2) 
        # output = self.spatial_upsample(out3)

        output = self.final_layers(x).squeeze(2)


        return output
    
class UNet3D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet3D, self).__init__()


        # Load pre-trained ResNet34 model
        resnet = eval(f"torchvision.models.{config.cnn_backbone}(pretrained ={config.resnet_pretrained})")
        resnet_layers = list(resnet.children())
                # List of supported layer types for conversion
        supported_layers = (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d)

        self.resnet_layers = nn.ModuleList()

        for layer in resnet_layers:
            if isinstance(layer, supported_layers):
                if isinstance(layer, nn.Conv2d):
                    # Convert Conv2D to Conv3D
                    conv3d = nn.Conv3d(
                        in_channels=12,
                        out_channels=layer.out_channels,
                        kernel_size=(1, layer.kernel_size[0], layer.kernel_size[1]),
                        stride=(1, layer.stride[0], layer.stride[1]),
                        padding=(0, layer.padding[0], layer.padding[1]),
                        bias=(layer.bias is not None)
                    )
                    # Adapt weights
                    original_weights = layer.weight.data  # Shape: (out_channels, 3, H, W)
                    
                    # Step 1: Expand RGB weights to 8 spectral bands
                    expanded_weights = original_weights.repeat(1, 3, 1, 1)[:, :12, :, :]  # Shape: (out_channels, 12, H, W)

                    # Step 2: Add depth axis (1, 1, H, W)
                    expanded_weights = expanded_weights.unsqueeze(2)  # Shape: (out_channels, 12, 1, H, W)

                    conv3d.weight.data = expanded_weights
                    if layer.bias is not None:
                        conv3d.bias.data = layer.bias.data
                    self.resnet_layers.append(conv3d)
                elif isinstance(layer, nn.BatchNorm2d):
                    # Convert BatchNorm2D to BatchNorm3D
                    bn3d = nn.BatchNorm3d(
                        num_features=layer.num_features,
                        eps=layer.eps,
                        momentum=layer.momentum,
                        affine=layer.affine,
                        track_running_stats=layer.track_running_stats
                    )
                    bn3d.weight.data = layer.weight.data
                    bn3d.bias.data = layer.bias.data
                    bn3d.running_mean = layer.running_mean
                    bn3d.running_var = layer.running_var
                    self.resnet_layers.append(bn3d)
                elif isinstance(layer, nn.ReLU):
                    self.resnet_layers.append(nn.ReLU(inplace=True))
                elif isinstance(layer, nn.MaxPool2d):
                    # Convert MaxPool2D to MaxPool3D
                    pool3d = nn.MaxPool3d(
                        kernel_size=(1, layer.kernel_size, layer.kernel_size),
                        stride=(1, layer.stride, layer.stride),
                        padding=(0, layer.padding, layer.padding)
                    )
                    self.resnet_layers.append(pool3d)
            else:
                print(f"Skipping unsupported layer: {type(layer)}")


        # Encoder
        self.encoder1 = nn.Sequential(
            # Block 1
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(128),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        # Block 2
        self.encoder2 = nn.Sequential(

            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(256),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.encoder3 = nn.Sequential(
            # Block 3
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(512),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.Featuredifference1 = FeatureDifferenceModule(1024)
        self.Featuredifference2 = FeatureDifferenceModule(512)
        self.Featuredifference3 = FeatureDifferenceModule(256)


        # Intermediate Convolutional Layer
        self.intermediate_conv = nn.Conv3d(1536, 1536, kernel_size=3, padding=1)
        self.intermediate_relu = nn.ReLU(inplace=True)

        # Decoder
        self.decoder1 = nn.Sequential(
            # Upsampling Block 1
            nn.ConvTranspose3d(3072, 1024, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(512)
        )
        self.decoder2 = nn.Sequential(
            # Upsampling Block 2
            nn.ConvTranspose3d(1280, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(128)
        )
        self.decoder3 = nn.Sequential(
            # Upsampling Block 3
            nn.ConvTranspose3d(512, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(64)
        )

        # Final 1x1 convolution for binary classification
        self.final_3conv = nn.ConvTranspose3d(64, num_classes, kernel_size=2,stride=2)
        # Final 1x1 convolution for binary classification
        self.final_2conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, data):
        # Encoder
        x = data
        
        for i in range(3):
            x= self.resnet_layers[i](x)
 

        x1 = self.encoder1(x)
        
        x2 = self.encoder2(x1)
        
        x3 = self.encoder3(x2)
        

  

        # Intermediate Convolutional Layer
        x = self.intermediate_conv(x3)
        x = self.intermediate_relu(x)

        # Decoder

        x = self.decoder1(torch.cat([x, x3], dim=1))
        x = self.decoder2(torch.cat([x, x2], dim=1))
        x = self.decoder3(torch.cat([x, x1], dim=1))


        # Final 1x1 convolution for binary classification
        output1 = self.final_3conv(x)

        # Stack the spectral information along dimension 1
        stacked_x = output1.view(1, -1, 128, 128)

        output = self.final_2conv(stacked_x) 

        return output
    
class SqueezeExcitation_(nn.Module):
   
    def __init__(self, num_channels, reduction_ratio=16):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(SqueezeExcitation_, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(input_tensor)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1))

        return output_tensor  
        
class FeatureDifferenceModule_(nn.Module):
    def __init__(self, dim):
        super(FeatureDifferenceModule_, self).__init__()
        self.se = SqueezeExcitation_(dim)  # Squeeze-and-Excitation block
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
          )

    def forward(self, x3_1, x3_2):
        # Operation 1: Absolute difference
        p_i1 = torch.abs(x3_1 - x3_2)

        # Operation 2: Concatenate features, apply SE and Convolution
        concatenated = torch.cat((x3_1, x3_2), dim=1)  # Concatenate along channel dimension
        se_output = self.se(concatenated)  # Squeeze-and-Excitation
        p_i2 = self.conv(se_output)  # Convolution

        # Fuse features
        FFi = torch.cat((p_i1, p_i2), dim=1)  # Concatenate along channel dimension

        return FFi    

class UNet2D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet2D, self).__init__()

        # Load pre-trained ResNet34 model
        resnet = eval(f"torchvision.models.{config.cnn_backbone}(pretrained={config.resnet_pretrained})")
        resnet_layers = list(resnet.children())

        # List of supported layer types for conversion
        supported_layers = (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d)
        self.resnet_layers = nn.ModuleList()

        for layer in resnet_layers:
            if isinstance(layer, supported_layers):
                if isinstance(layer, nn.Conv2d):
                    # Adjust the first Conv2D layer for 8 spectral bands
                    if layer.in_channels == 3:  # This ensures only the first layer is modified
                        conv2d = nn.Conv2d(
                            in_channels=8,
                            out_channels=layer.out_channels,
                            kernel_size=layer.kernel_size,
                            stride=layer.stride,
                            padding=layer.padding,
                            bias=(layer.bias is not None)
                        )
                        
                        # Adapt weights for 8 input bands
                        original_weights = layer.weight.data  # Shape: (out_channels, 3, H, W)
                        adjusted_weights = original_weights.mean(dim=1, keepdim=True).repeat(1, 8, 1, 1)  # Shape: (out_channels, 8, H, W)
                        conv2d.weight.data = adjusted_weights

                        if layer.bias is not None:
                            conv2d.bias.data = layer.bias.data

                        self.resnet_layers.append(conv2d)
                    else:
                        self.resnet_layers.append(layer)  # Add unchanged Conv2D layers
                elif isinstance(layer, nn.BatchNorm2d):
                    # Add BatchNorm2D layers as is
                    self.resnet_layers.append(layer)
                elif isinstance(layer, nn.ReLU):
                    # Add ReLU layers as is
                    self.resnet_layers.append(nn.ReLU(inplace=True))
                elif isinstance(layer, nn.MaxPool2d):
                    # Add MaxPool2D layers as is
                    self.resnet_layers.append(layer)
            else:
                print(f"Skipping unsupported layer: {type(layer)}")


        # Encoder
        self.encoder1 = nn.Sequential(
            # Block 1
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 2
        self.encoder2 = nn.Sequential(

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.encoder3 = nn.Sequential(
            # Block 3
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.Featuredifference1 = FeatureDifferenceModule_(1024)
        self.Featuredifference2 = FeatureDifferenceModule_(512)
        self.Featuredifference3 = FeatureDifferenceModule_(256)


        # Intermediate Convolutional Layer
        self.intermediate_conv = nn.Conv2d(1536, 1536, kernel_size=3, padding=1)
        self.intermediate_relu = nn.ReLU(inplace=True)

        # Decoder
        self.decoder1 = nn.Sequential(
            # Upsampling Block 1
            nn.ConvTranspose2d(3072, 1024, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        )
        self.decoder2 = nn.Sequential(
            # Upsampling Block 2
            nn.ConvTranspose2d(1280, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        )
        self.decoder3 = nn.Sequential(
            # Upsampling Block 3
            nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )

        # Final 1x1 convolution for binary classification
        #self.final_3conv = nn.Conv3d(32, num_classes, kernel_size=1)
        # Final 1x1 convolution for binary classification
        self.final_2conv = nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2)

    def forward(self, data1, data2):
        # Encoder
        
        x1 = data1
        x2 = data2
        for i in range(3):
            x1= self.resnet_layers[i](x1)

        for i in range(3):
            x2= self.resnet_layers[i](x2)    

        x1_1 = self.encoder1(x1)
        x1_2 = self.encoder1(x2)
        x2_1 = self.encoder2(x1_1)
        x2_2 = self.encoder2(x1_2)
        x3_1 = self.encoder3(x2_1)
        x3_2 = self.encoder3(x2_2)

        # Feature difference (bottleneck)
        x3 = self.Featuredifference1(x3_1, x3_2)
        x2 = self.Featuredifference2(x2_1, x2_2)
        x1 = self.Featuredifference3(x1_1, x1_2)        

        # Intermediate Convolutional Layer
        x = self.intermediate_conv(x3)
        x = self.intermediate_relu(x)

        # Decoder

        x = self.decoder1(torch.cat([x, x3], dim=1))
        x = self.decoder2(torch.cat([x, x2], dim=1))
        x = self.decoder3(torch.cat([x, x1], dim=1))


        # Final 1x1 convolution for binary classification
        output = self.final_2conv(x)

        # Stack the spectral information along dimension 1
        #stacked_x = output1.view(1, -1, 128, 128)

        #output = self.final_2conv(stacked_x) 

        return output    
    
class Res_AttentionBlock2D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Res_AttentionBlock2D, self).__init__()

        self.F_l = F_l
        self.F_g = F_g
        self.F_int = F_int

        self.relu = nn.ReLU(inplace=True)


    def forward(self, g, x):
            # Determine the temporal dimension from input tensors
        temporal_dim = g.shape[2]
        
        if temporal_dim > 1:
            kernel = 3 
            
        else:
            kernel = 1
        
        padding_size = kernel//2 

        #Determine device from input tensors
        device = x.device

        # Define convolutions with the determined kernel size
        W_x = nn.Sequential(
            nn.Conv3d(self.F_l, self.F_int, kernel_size=(1, 2, 2), stride=(2, 2, 2), padding=(0,0,0), bias=True),
            nn.BatchNorm3d(self.F_int)
        ).to(device)

        W_g = nn.Sequential(
            nn.Conv3d(self.F_g, self.F_int, kernel_size=(1, 1, 1), stride=1, padding=(0,0,0), bias=True),
            nn.BatchNorm3d(self.F_int)
        ).to(device)

        psi = nn.Sequential(
            nn.Conv3d(self.F_int, 1, kernel_size=(1, 1, 1), stride=1, padding=(0,0,0), bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        ).to(device)

        # Forward pass with the dynamically defined convolutions
        g1 = W_g(g)
        x1 = W_x(x)
        psi_out = self.relu(g1 + x1)
        psi_out = psi(psi_out)

        #Upsample psi_out by a factor of 2 along spatial dimensions
        psi_out = F.interpolate(psi_out, scale_factor=(2,2,2), mode='trilinear', align_corners=False)
        
        return x * psi_out  # Applying attention to x instead of g   
    
     

class ConvWindowAttention_(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super(ConvWindowAttention_, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        self.qkv_conv = nn.Conv1d(dim, dim * 3, kernel_size=1)
        self.out_conv = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        B, N, C = x.shape  # Expecting 4D input shape
        assert C == self.dim, f"Expected input channel {self.dim}, but got {C}."
        

        # Reshape input to (B, N, C) for compatibility with Conv1d
        #x = x.view(B, C, N).permute(0, 2, 1)  # Shape: (B, N, C)

        # Project to QKV space and reshape
        qkv = self.qkv_conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)

        # Separate q, k, v
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Scaled dot-product attention
        attention_scores = (q @ k.transpose(-2, -1)) * self.scale
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_output = (attention_scores @ v)

        # Reshape output back to (B, C, D, H, W)
        attention_output = attention_output.transpose(1, 2).reshape(B, N, C)
        x = self.out_conv(attention_output.permute(0, 2, 1)).permute(0, 2, 1)
        
        return x


class ConvWindowTransformerBlock_(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super(ConvWindowTransformerBlock_, self).__init__()
        self.attention_layer = ConvWindowAttention_(dim, num_heads, window_size)
        self.conv_layer = nn.Sequential(
            nn.Conv1d(dim, dim * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim * 4, dim, kernel_size=3, padding=1)
        )

    def forward(self, x):
        attention_output = self.attention_layer(x)
        x = x + attention_output

        conv_output = self.conv_layer(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + conv_output
        return x

class PatchConvMerging(nn.Module):
    def __init__(self, dim):
        super(PatchConvMerging, self).__init__()
        self.dim = dim
        self.conv = nn.Conv3d(dim * 8, dim * 2, kernel_size=1)

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        assert N == D * H * W, "Input feature has incorrect spatial dimensions"

        x = x.view(B, D, H, W, C)
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, 0::2, :]
        x4 = x[:, 0::2, 0::2, 1::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=-1)
        x = x.view(B, -1, 8 * C)

        x = self.conv(x.permute(0, 2, 1).view(B, -1, D // 2, H // 2, W // 2))
        
        return x

class CrossAttentionFusion_(nn.Module):
    def __init__(self, dim):
        super(CrossAttentionFusion_, self).__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.fusion_conv = nn.Conv2d(2 * dim, dim, kernel_size=1)


    def forward(self, transformer_features, cnn_features):
        # Reshape features
        B, C, H, W = cnn_features.shape
        cnn_features = cnn_features.view(B, C, H * W).transpose(1, 2)  # Shape: (B, N, C)
        transformer_features = transformer_features.view(B, C, H * W).transpose(1, 2)  # Shape: (B, N, C)
        
        # Project to Q, K, V
        key = self.key_proj(cnn_features)  # CNN output as Key
        value = self.value_proj(cnn_features)  # CNN output as Value
        query = self.query_proj(transformer_features)  # Transformer output as Query


        # Compute attention scores
        attention_scores = torch.bmm(query, key.transpose(1, 2)) / (C ** 0.5)
        attention_weights = self.softmax(attention_scores)  # Shape: (B, N, N)

        # Weighted sum of values (attention mechanism)
        attended_features = torch.bmm(attention_weights, value)

        # Reshape back to 3D form and add with transformer features for final fusion
        attended_features = attended_features.transpose(1, 2).view(B, C, H, W)
        transformer_features = transformer_features.transpose(1, 2).view(B, C, H, W)
        # Modify the final fusion step
        concatenated_features = torch.cat((transformer_features, attended_features), dim=1)
        fused_features = self.fusion_conv(concatenated_features)  # Additional convolution after concatenation

        #fused_features = transformer_features + attended_features  # Fusion by addition
        return fused_features

# Modify your encoder to use the CrossAttentionFusion module
class AdaptiveSwinTransformerEncoder_(nn.Module):
    def __init__(self, in_channels, base_window_size=7, num_heads=[4, 8, 16]):
        super(AdaptiveSwinTransformerEncoder_, self).__init__()
        
        # Define CNN and Transformer layers
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.transformer_block1 = ConvWindowTransformerBlock_(dim=32, num_heads=num_heads[0], window_size=base_window_size)
        self.cross_attention1 = CrossAttentionFusion_(dim=32)

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.transformer_block2 = ConvWindowTransformerBlock_(dim=64, num_heads=num_heads[1], window_size=base_window_size)
        self.cross_attention2 = CrossAttentionFusion_(dim=64)

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.transformer_block3 = ConvWindowTransformerBlock_(dim=128, num_heads=num_heads[2], window_size=base_window_size)
        self.cross_attention3 = CrossAttentionFusion_(dim=128)

    def forward(self, x):
        # Level 1 features
        conv_x1 = self.conv_block1(x)
        B, C, H, W = conv_x1.shape
        x1_flattened = conv_x1.view(B, C,  H * W).transpose(1, 2)
        trans_x1 = self.transformer_block1(x1_flattened).transpose(1, 2).view(B, C, H, W)
        x1 = self.cross_attention1(trans_x1, conv_x1)

        # Level 2 features
        conv_x2 = self.conv_block2(x1)
        B, C, H, W = conv_x2.shape
        x2_flattened = conv_x2.view(B, C,  H * W).transpose(1, 2)
        trans_x2 = self.transformer_block2(x2_flattened).transpose(1, 2).view(B, C, H, W)
        x2 = self.cross_attention2(trans_x2, conv_x2)

        # Level 3 features
        conv_x3 = self.conv_block3(x2)
        B, C, H, W = conv_x3.shape
        x3_flattened = conv_x3.view(B, C,  H * W).transpose(1, 2)
        trans_x3 = self.transformer_block3(x3_flattened).transpose(1, 2).view(B, C, H, W)
        x3 = self.cross_attention3(trans_x3, conv_x3)

        return x1, x2, x3 
    
  
    


class DynamicAttentionUNet2D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DynamicAttentionUNet2D, self).__init__()

        # Encoder with Adaptive Swin Transformer
        self.adaptive_swin_encoder = AdaptiveSwinTransformerEncoder_(in_channels)
        self.Featuredifference1 = FeatureDifferenceModule_(256)
        self.Featuredifference2 = FeatureDifferenceModule_(128)
        self.Featuredifference3 = FeatureDifferenceModule_(64)

        # Intermediate Convolutional Layer
        self.intermediate_conv = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.intermediate_activation = nn.ReLU(inplace=True)

        # Decoder (Remains unchanged)
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(384, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8)
        )
        
        #input 32
        self.convinput = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        # Attention Blocks (Unchanged)
        self.attention3 = Res_AttentionBlock2D(F_g=128, F_l=64, F_int=128)
        self.attention2 = Res_AttentionBlock2D(F_g=64, F_l=32, F_int=64)
        self.attention1 = Res_AttentionBlock2D(F_g=32, F_l=16, F_int=32)

        # Final 1x1 convolution for binary classification
        #self.final_3conv = nn.Conv3d(8, num_classes, kernel_size=1)
        self.final_2conv = nn.Conv2d(8, num_classes, kernel_size=1)

        
    def forward(self, data1, data2 ):
        # Encoder
        encoder_outputs1 = self.adaptive_swin_encoder(data1)
        encoder_outputs2 = self.adaptive_swin_encoder(data2)
        x1_1, x2_1, x3_1 = encoder_outputs1
        x1_2, x2_2, x3_2 = encoder_outputs2

        # Feature difference (bottleneck)
        x3 = self.Featuredifference1(x3_1, x3_2)
        x2 = self.Featuredifference2(x2_1, x2_2)
        x1 = self.Featuredifference3(x1_1, x1_2)

        # Intermediate Convolutional Layer
        x = self.intermediate_conv(x3)
        x = self.intermediate_activation(x)

        # Decoder with Attention Gates
        xd = self.decoder3[0:1](x)
        #x3 = self.attention3(g=x, x=x2)
        x = torch.cat([xd, x2], dim=1)
        x = self.decoder3[2:](x)

        xd = self.decoder2[0:1](x)
        #x2 = self.attention2(g=x, x=x1)
        x = torch.cat([xd, x1], dim=1)
        x = self.decoder2[2:](x)

        xd = self.decoder1[0:1](x)
        #x0 = self.convinput(input1)
        #x1 = self.attention1(g=x, x=x0)
        #x = torch.cat([xd, x1], dim=1)
        x = self.decoder1[2:](xd)

        # Final 1x1 convolution for binary classification
        output = self.final_2conv(x)

        # Stack the spectral information along dimension 1
        #stacked_x = output1.view(1, -1, 128, 128)

        #output = self.final_2conv(stacked_x)

        return output    

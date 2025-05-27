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

        # Intermediate Conv
        self.intermediate_conv = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.intermediate_activation = nn.ReLU(inplace=True)

        # Decoder Blocks (Upsampling spatial dimensions only)
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose3d(1024, 512, kernel_size=(2, 2, 2), stride=(2, 2, 2)),  # 16x16 → 32x32
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(256)
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=(2, 2, 2), stride=(2, 2, 2)),  # 32x32 → 64x64
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(128)
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=(2, 2, 2), stride=(1, 2, 2)),   # 64x64 → 128x128
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(64)
        )

        # Final Layer: Collapse dim and upsample to 256x256
        self.collapse1 = nn.Conv3d(64, 32, kernel_size=(3, 1, 1))  # 2 → 1
        self.collapse2 = nn.Conv3d(32, 16, kernel_size=(3, 1, 1))  # 2 → 1
        self.collapse3 = nn.Conv3d(16, 8, kernel_size=(2, 1, 1))  # 2 → 1
        #self.temporal_collapse4 = nn.Conv3d(32, 32, kernel_size=(2, 1, 1))
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # 128x128 → 256x256
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1)            # Final classification layer
        )


        
    def forward(self, data):
        # Encoder
        encoder_outputs = self.adaptive_swin_encoder(data)
        
        x1, x2, x3 = encoder_outputs

        x = self.intermediate_conv(x3)
        x = self.intermediate_activation(x)

        x = self.decoder3(torch.cat([x, x3], dim=1))
        x = self.decoder2(torch.cat([x, x2], dim=1))
        x = self.decoder1(torch.cat([x, x1], dim=1))
        

        x = self.collapse1(x)  # (B, 32, 1, 128, 128)
        x = self.collapse2(x)
        x = self.collapse3(x)
        #x = self.temporal_collapse4(x)
        B, C, D, H, W = x.shape
        x_out = x.view(1, -1, H, W)              # (B, 32, 128, 128)
        output = self.final_upsample(x_out)     # (B, num_classes, 256, 256)
        #output = output.squeeze(1) 


        return output

import torch
import torch.nn as nn
import torch.nn.functional as F
from MNIST.models.layers import *



class UNet_res(nn.Module):
    """
    Copied and modified from https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py#L337
    """
    def __init__(self,
                 input_channels,
                 input_height,
                 ch,
                 output_channels=None,
                 ch_mult=(1,2,2),     #(1, 2, 4, 8),
                 num_res_blocks=2,
                 attn_resolutions=(16,),
                 dropout=0.,
                 resamp_with_conv=True,
                 act=nn.SiLU(),
                 normalize=group_norm,
                 self_attention=False
                 ):
        super().__init__()
        self.input_channels = input_channels
        self.input_height = input_height
        self.ch = ch
        self.output_channels = output_channels = input_channels if output_channels is None else output_channels
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.resamp_with_conv = resamp_with_conv
        self.act = act
        self.normalize = normalize
        self.self_attention = self_attention

        # init
        self.num_resolutions = num_resolutions = len(ch_mult)
        in_ht = input_height
        in_ch = input_channels
        temb_ch = ch * 4
        assert in_ht % 2 ** (num_resolutions - 1) == 0, "input_height doesn't satisfy the condition"

        # Timestep embedding
        self.temb_net = TimestepEmbedding(
            embedding_dim=ch,
            hidden_dim=temb_ch,
            output_dim=temb_ch,
            act=act,
        )

        # Downsampling
        self.begin_conv = conv2d(in_ch, ch)
        unet_chs = [ch]
        in_ht = in_ht
        in_ch = ch
        down_modules = []
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            block_modules = {}
            out_ch = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks):
                block_modules['{}a_{}a_block'.format(i_level, i_block)] = \
                    ResidualBlock(
                        in_ch=in_ch,
                        temb_ch=temb_ch,
                        out_ch=out_ch,
                        dropout=dropout,
                        act=act,
                        normalize=normalize,
                    )
                if self.self_attention and in_ht in attn_resolutions:
                    block_modules['{}a_{}b_attn'.format(i_level, i_block)] = SelfAttention(out_ch, normalize=normalize)
                unet_chs += [out_ch]
                in_ch = out_ch
            # Downsample
            if i_level != num_resolutions - 1:
                block_modules['{}b_downsample'.format(i_level)] = downsample(out_ch, with_conv=resamp_with_conv)
                in_ht //= 2
                unet_chs += [out_ch]
            # convert list of modules to a module list, and append to a list
            down_modules += [nn.ModuleDict(block_modules)]
        # conver to a module list
        self.down_modules = nn.ModuleList(down_modules)

        # Middle
        mid_modules = []
        mid_modules += [
            ResidualBlock(in_ch, temb_ch=temb_ch, out_ch=in_ch, dropout=dropout, act=act, normalize=normalize)]
        if self.self_attention:
            mid_modules += [SelfAttention(in_ch, normalize=normalize)]
        mid_modules += [
            ResidualBlock(in_ch, temb_ch=temb_ch, out_ch=in_ch, dropout=dropout, act=act, normalize=normalize)]
        self.mid_modules = nn.ModuleList(mid_modules)

        # Upsampling
        up_modules = []
        for i_level in reversed(range(num_resolutions)):
            # Residual blocks for this resolution
            block_modules = {}
            out_ch = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks + 1):
                block_modules['{}a_{}a_block'.format(i_level, i_block)] = \
                    ResidualBlock(
                        in_ch=in_ch + unet_chs.pop(),
                        temb_ch=temb_ch,
                        out_ch=out_ch,
                        dropout=dropout,
                        act=act,
                        normalize=normalize)
                if self.self_attention and in_ht in attn_resolutions:
                    block_modules['{}a_{}b_attn'.format(i_level, i_block)] = SelfAttention(out_ch, normalize=normalize)
                in_ch = out_ch
            # Upsample
            if i_level != 0:
                block_modules['{}b_upsample'.format(i_level)] = upsample(out_ch, with_conv=resamp_with_conv)
                in_ht *= 2
            # convert list of modules to a module list, and append to a list
            up_modules += [nn.ModuleDict(block_modules)]
        # conver to a module list
        self.up_modules = nn.ModuleList(up_modules)
        assert not unet_chs

        # End
        self.end_conv = nn.Sequential(
            normalize(in_ch),
            self.act,
            conv2d(in_ch, output_channels, init_scale=0.),
        )

    # noinspection PyMethodMayBeStatic
    def _compute_cond_module(self, module, x, temp):
        for m in module:
            x = m(x, temp)
        return x

    # noinspection PyArgumentList,PyShadowingNames
    def forward(self, x, temp):
        # Init
        B, C, H, W = x.size()

        # Timestep embedding
        temb = self.temb_net(temp)
        assert list(temb.shape) == [B, self.ch * 4]

        # Downsampling
        hs = [self.begin_conv(x)]
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            block_modules = self.down_modules[i_level]
            for i_block in range(self.num_res_blocks):
                resnet_block = block_modules['{}a_{}a_block'.format(i_level, i_block)]
                h = resnet_block(hs[-1], temb)
                if h.size(2) in self.attn_resolutions:
                   attn_block = block_modules['{}a_{}b_attn'.format(i_level, i_block)]
                   h = attn_block(h, temb)
                hs.append(h)
            # Downsample
            if i_level != self.num_resolutions - 1:
                downsample = block_modules['{}b_downsample'.format(i_level)]
                hs.append(downsample(hs[-1]))

        # Middle
        h = hs[-1]
        h = self._compute_cond_module(self.mid_modules, h, temb)

        # Upsampling
        for i_idx, i_level in enumerate(reversed(range(self.num_resolutions))):
            # Residual blocks for this resolution
            block_modules = self.up_modules[i_idx]
            for i_block in range(self.num_res_blocks + 1):
                resnet_block = block_modules['{}a_{}a_block'.format(i_level, i_block)]
                h = resnet_block(torch.cat([h, hs.pop()], axis=1), temb)
                if h.size(2) in self.attn_resolutions:
                   attn_block = block_modules['{}a_{}b_attn'.format(i_level, i_block)]
                   h = attn_block(h, temb)
            # Upsample
            if i_level != 0:
                upsample = block_modules['{}b_upsample'.format(i_level)]
                h = upsample(h)
        assert not hs

        # End
        h = self.end_conv(h)
        assert list(h.size()) == [x.size(0), self.output_channels, x.size(2), x.size(3)]
        return h



class UNet_simple(nn.Module):
    """
    Copied and modified from https://github.com/LabForComputationalVision/bias_free_denoising/blob/master/models/unet.py
    """
    
    def __init__(self,residual_connection=False,n_feature=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1,n_feature,5,padding=2)
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_feature, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(n_feature+64,n_feature,3,padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_feature, eps=1e-6, affine=True)
        self.conv3 = nn.Conv2d(n_feature+64,n_feature*2,3,stride=2,padding=1)
        self.norm3 = nn.GroupNorm(num_groups=32, num_channels=n_feature*2, eps=1e-6, affine=True)
        self.conv4 = nn.Conv2d(n_feature*2+64,n_feature*2,4,padding=1)
        self.norm4 = nn.GroupNorm(num_groups=32, num_channels=n_feature*2, eps=1e-6, affine=True)
        self.conv5 = nn.Conv2d(n_feature*2+64,n_feature*2,3,dilation=2,padding=2)
        self.norm5 = nn.GroupNorm(num_groups=32, num_channels=n_feature*2, eps=1e-6, affine=True)
        self.conv6 = nn.Conv2d(n_feature*2+64,n_feature*2,3,dilation=4,padding=4)
        self.norm6 = nn.GroupNorm(num_groups=32, num_channels=n_feature*2, eps=1e-6, affine=True)
        self.conv7 = nn.ConvTranspose2d(n_feature*2+64,n_feature*2,4,stride=2,padding=0)
        self.norm7 = nn.GroupNorm(num_groups=32, num_channels=n_feature*2, eps=1e-6, affine=True)
        self.conv8 = nn.Conv2d(n_feature*3+64,n_feature,3,padding=1)
        self.norm8 = nn.GroupNorm(num_groups=32, num_channels=n_feature, eps=1e-6, affine=True)
        self.conv9 = nn.Conv2d(n_feature+64,1,5,padding=2,bias=False)
        self.sigma_emb_1 = nn.Linear(16,32)
        self.sigma_emb_2 = nn.Linear(32,64)
        self.residual_connection = residual_connection


    def forward(self,x,sigma):
        pad_right = x.shape[-2]%2
        pad_bottom = x.shape[-1]%2
        padding = nn.ZeroPad2d((0,pad_bottom,0,pad_right))
        x = padding(x)
        
        sigma = get_sinusoidal_positional_embedding(sigma,16)
        sigma = F.silu(self.sigma_emb_1(sigma))
        sigma = F.silu(self.sigma_emb_2(sigma))


        out = F.silu(self.norm1(self.conv1(x)))
        sigma_shape = sigma.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,out.shape[-2],out.shape[-1])
        out = torch.cat([out,sigma_shape],dim=1)

        out_saved = F.silu(self.norm2(self.conv2(out)))
        sigma_shape = sigma.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,out_saved.shape[-2],out_saved.shape[-1])
        out_saved = torch.cat([out_saved,sigma_shape],dim=1)    
        
        out = F.silu(self.norm3(self.conv3(out_saved)))
        sigma_shape = sigma.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,out.shape[-2],out.shape[-1])
        out = torch.cat([out,sigma_shape],dim=1)    
        out = F.silu(self.norm4(self.conv4(out)))
        sigma_shape = sigma.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,out.shape[-2],out.shape[-1])
        out = torch.cat([out,sigma_shape],dim=1)    


        out = F.silu(self.norm5(self.conv5(out)))
        sigma_shape = sigma.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,out.shape[-2],out.shape[-1])
        out = torch.cat([out,sigma_shape],dim=1)  
        
        out = F.silu(self.norm6(self.conv6(out)))
        sigma_shape = sigma.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,out.shape[-2],out.shape[-1])
        out = torch.cat([out,sigma_shape],dim=1)   
        out = F.silu(self.norm7(self.conv7(out)))
        out = torch.cat([out,out_saved],dim = 1)

        out = F.silu(self.norm8(self.conv8(out)))
        sigma_shape = sigma.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,out.shape[-2],out.shape[-1])
        out = torch.cat([out,sigma_shape],dim=1) 
        out = self.conv9(out)

        if self.residual_connection:
            out = x - out
        return out 

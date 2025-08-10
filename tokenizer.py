import dac
import torch


class Tokenizer:
    def __init__(self, format="dac", device=torch.device("cpu")):
        if format = "dac":
            self.format = "dac"
            self.num_codebooks = 9
            self.model = dac.DAC.load(dac.utils.download(model_type="44khz")).to(device).eval()
    
    def from_codes(self, codes):

        if format = "dac":
            return self.model.quantizer.from_codes(codes)[0]
    
    def decode_code(self, i, code):

        if format = "dac":
            temp_emb = self.model.quantizer.quantizers[i].decode_code(code)
            temp_emb = self.model.quantizer.quantizers[i].out_proj(temp_emb)
            return temp

    def code_to_audio(self, codes):

        if format = "dac":
            o_emb = self.model.quantizer.from_codes(codes)[0]
            return self.model.decode(o_emb)
    
    def emb_toaudio(self, emb):

        if format = "dac":
            return self.model.decode(emb)


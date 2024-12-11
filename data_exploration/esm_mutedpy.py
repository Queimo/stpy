import esm.pretrained
from stpy.embeddings.embedding import Embedding
import pickle
from mutedpy.utils.protein_operator import ProteinOperator
import torch
import os
from esm.models.esm3 import ESM3
from esm.pretrained import load_local_model
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.pretrained import LOCAL_MODEL_REGISTRY
from huggingface_hub import login
from transformers import AutoTokenizer, EsmModel
from transformers import AutoTokenizer, AutoModelForMaskedLM

class ESMEmbedding(Embedding):

    def __init__(self, name = 'esm-1v', device = 'cpu', preloaded = None, save_location = None, mean = False, length = 91):
        self.name = name
        if name == 'esm-1v':
            self.model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_5()
            self.batch_converter = alphabet.get_batch_converter()
            self.model.eval()

        elif name == 'esm-2':
            login("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")

            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
            self.model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")
            self.batch_converter = lambda sequences : self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=True)

            # self.model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            # self.batch_converter = alphabet.get_batch_converter()
            self.model.eval()

        elif name == 'esm-3':
            login("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")
            client = ESM3.from_pretrained("esm3_sm_open_v1", device=torch.device("cpu"))  # or "cpu"
            def query(seq):
                protein = ESMProtein(sequence=(seq))
                protein_tensor = client.encode(protein)
                output = client.forward_and_sample(protein_tensor, SamplingConfig(return_per_residue_embeddings=True))
                return output.per_residue_embedding
            self.model = lambda seq: query(seq)
        else:
            raise AssertionError("Not Implemented.")

        self.mean = mean
        if mean:
            self.m = 1280
        else:
            self.m = 1280*length
        self.device = device
        self.feature_names = [name + "_" + str(i) for i in range(self.m)]

        if preloaded is not None:
            if os.path.exists(preloaded):
                self.dict = pickle.load(open(preloaded, 'rb'))
            else:
                self.dict = {}
        else:
            self.dict = {}


    def dump_embeddings(self, name):
        pickle.dump(self.dict, open(name,"wb"))

    def load_embeddings(self,name):
        self.dict = pickle.load(open(name,"rb"))

    def get_m(self):
        return self.m

    def embed_seq_all_new(self, x):
        tokenized_inputs = self.batch_converter(x)
        with torch.no_grad():
            outputs = self.model(**tokenized_inputs)
            #results = self.model(tokenized_inputs, repr_layers=[33], return_contacts=False)
            #token_representations = results["representations"][33]
            token_representations = outputs.last_hidden_state
            if self.mean:
                z = torch.mean(token_representations, dim = 1)
            else:
                z = token_representations
        return z

    def embed_seq_esm3(self, x, device=None, batch_size=10, verbose=False):
        keys = self.dict.keys()
        out = []
        for s in x:
            if s in keys:
                out.append(self.dict[s])
            else:
                z = self.model(s)
                if self.mean:
                    z = z.mean(0)
                else:
                    pass
                out.append(z)
                self.dict[s] = z
        return torch.stack(out).double().to('cpu')

    def embed_seq(self,x, device = None, batch_size = 10, verbose = False):
        if device is None:
            device = self.device

        if self.name == 'esm-3':
            return self.embed_seq_esm3(x, device = device, batch_size = batch_size, verbose = verbose)

        out = []
        keys = self.dict.keys()
        self.model.to(device)
        for s in x:
            # if already embedded use the one
            if s in keys:
                out.append(self.dict[s])
            # embed with the pretrained model
            else:
                #batch_labels, batch_strs, batch_tokens = self.batch_converter([("dummy", s)])
                with torch.no_grad():
                    # results = self.model(batch_tokens.to(device), repr_layers=[33], return_contacts=False)
                    # token_representations = results["representations"][33]
                    tokenized_inputs = self.batch_converter([s])
                    outputs = self.model(**tokenized_inputs.to(device), output_hidden_states=True)
                    # results = self.model(tokenized_inputs, repr_layers=[33], return_contacts=False)
                    # token_representations = results["representations"][33]
                    token_representations = outputs.hidden_states[-1]#outputs.last_hidden_state

                    if self.mean:
                        z = token_representations[0, :, :].mean(0)
                    else:
                        z = token_representations[0, :, :]
                self.dict[s] = z
                out.append(z)
        return torch.stack(out).double().to('cpu')

    def embed(self, x):
        out = []
        keys = self.dict.keys()
        P = ProteinOperator()
        # calculate sequences
        for xx in x:
            s = P.translate_seq_to_variant(xx)
            # if already embedded use the one
            if s in keys:
                out.append(self.dict[s])
            # embed with the pretrained model
            else:
                batch_labels, batch_strs, batch_tokens = self.batch_converter([("dummy",s)])
                with torch.no_grad():
                    results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
                    token_representations = results["representations"][33]
                    if self.mean:
                        z = token_representations[0,:,:].mean(0)
                    else:
                        z = token_representations[0, :, :]
                self.dict[s] = z
                out.append(z)
        return torch.stack(out).double()

if  __name__ == '__main__':
    import time
    from mutedpy.experiments.kemp.kemp_loader import *
    x,y ,dts = load_first_round()


    seq_list = dts['variant'].values.tolist()[0:2]


    embedding = ESMEmbedding('esm-2',mean=True)

    # t0 = time.time()
    # Phi = embedding.embed_seq_all_new(seq_list)
    # t1 = time.time()
    # print (t1-t0)
    # print (Phi.size())

    t0 = time.time()
    Phi = embedding.embed_seq(seq_list)
    t1 = time.time()
    print(t1 - t0)
    print (Phi.size())


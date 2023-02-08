from thop import profile
from thop import clever_format
import argparse
import torch
from model.discriminator import FCDiscriminator
from model.discriminator_dsc import DSCDiscriminator

def main():
   parser = argparse.ArgumentParser()
   parser.add_argument('--model', type=str, default="", help='The model to use. FCD to use FCDiscriminator or DSC to use DSCDiscriminator')
   args = parser.parse_args()

   m=""

   input = torch.randn(1, 19, 1024, 512)
   if (args.model=="FCD"):
    model= FCDiscriminator(19)
    m="FCDiscriminator"
   else:
    m="DSCDiscriminator" 
    model= DSCDiscriminator(19) 

   #profile restituisce il numero di operazioni a+(bxc) che però essendo due operazioni va
   #moltiplicato per due per ottenre il reale numero di operazioni
   flops,params = profile(model, inputs=(input, ))
   Totflops=2*flops
   
   flops, params = clever_format([flops, params], "%.3f")
   
   print("model: "+m)
   print("flops : "+str(Totflops))
   print("parms : "+params)
   

if __name__ == '__main__':
    main()

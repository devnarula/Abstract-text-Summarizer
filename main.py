# Importing dependencies from transformers
from lib2to3.pgen2 import token
from transformers import PegasusForConditionalGeneration
from transformers import PegasusTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
#Load Model
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
text = """
The treatment landscape of acute myeloid leukemia (AML) is evolving, with promising therapies entering clinical translation, yet patient responses remain heterogeneous, and biomarkers for tailoring treatment are lacking. To understand how disease heterogeneity links with therapy response, we determined the leukemia cell hierarchy makeup from bulk transcriptomes of more than 1,000 patients through deconvolution using single-cell reference profiles of leukemia stem, progenitor and mature cell types. Leukemia hierarchy composition was associated with functional, genomic and clinical properties and converged into four overall classes, spanning Primitive, Mature, GMP and Intermediate. Critically, variation in hierarchy composition along the Primitive versus GMP or Primitive versus Mature axes were associated with response to chemotherapy or drug sensitivity profiles of targeted therapies, respectively. A seven-gene biomarker derived from the Primitive versus Mature axis was associated with response to 105 investigational drugs. Cellular hierarchy composition constitutes a novel framework for understanding disease biology and advancing precision medicine in AML.
"""
# Create tokens - number representation of our text

# Sumamrize
# **tokens are our tokens

#Decode summary
# print(tokenizer.decode(summary[0]))
def compute(text):
    tokens = tokenizer(text, truncation = True, padding='longest', return_tensors='pt')
    summary = model.generate(**tokens)
    return tokenizer.decode(summary[0])


#---------------------api implementation-------------------
app = FastAPI()
class item(BaseModel):
    text: str
@app.post('/summarize/')
def summarize(Item: item):
    return compute(item.text)
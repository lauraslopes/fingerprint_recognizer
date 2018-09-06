Relatório trabalho 3 Visão Computacional
Aluna: Laura Silva Lopes, GRR20163048

Os caminhos (linha 438) até o diretório do dataset Lindex101 está como:
dataset_path = './Lindex101'

Estão implementados e executando corretamente no código:
-Fingerprint enhancement
-Computation of Orientation Map
-Region of interest detection
-Singular point detection (Poincaré index)
-Fingerprint Type Classification
-Thining
-Minutiae Extraction
-Pattern Matching

Para utilizar o dataset Rindex28, o processo é o mesmo. É necessário apenas, trocar o caminho da linha 438 para o diretório
Rindex28.  

Bugs encontrados:
Após passar por um processo de refinamento e remoção de falsas mínucias (spirious), algumas mínucias são removidas "por engano",
e outras, que não deviriam ser caracterizadas como minúcias, não são removidas.
O algoritmo de busca por pontos singulares nas imagens encontra muito mais pontos singulares do que realmente tem. Portanto,
a maioria das impressões digitais são classificadas como espiral (Whorl).

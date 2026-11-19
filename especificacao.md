UNIVERSIDADE FEDERAL DE MINAS GERAIS
Instituto de Ciências Exatas
Departamento de Ciência da Computação
DCC207 – Algoritmos 2
Prof. Renato Vimieiro
Trabalho Prático 2 - Análise Empírica e Otimização para o Caixeiro Viajante
Objetivos
Nesse trabalho, abordaremos os aspectos práticos da implementação e avaliação de algoritmos
para solucionar problemas difíceis (NP-Difíceis). Avaliaremos não apenas a construção sintática
do código, mas principalmente a auditoria de algoritmos, modelagem estatística
de desempenho e análise de sensibilidade aplicada ao Problema do Caixeiro Viajante
(TSP) Geométrico.
O objetivo principal é que os alunos desenvolvam um modelo crítico sobre as estruturas de
dados utilizadas, os gargalos teóricos versus práticos, e como as características topológicas dos
dados impactam a qualidade de soluções exatas e aproximadas. O uso de ferramentas de
Inteligência Artificial (LLMs) é permitido, mas desloca a carga de trabalho da codificação
primária para a instrumentação, refatoração de abstrações ineficientes e argumentação
científica profunda sobre o comportamento dos algoritmos.
Tarefas
Os alunos deverão auditar, refatorar e otimizar implementações dos seguintes algoritmos para
solucionar o TSP:
 .
O trabalho poderá ser desenvolvido em grupos de até 3 pessoas.
Instrumentação, Modelagem e Avaliação
Em seguida, os alunos deverão avaliar o desempenho dos algoritmos com instâncias do conjunto
de teste, com tempo de processamento limitado a 30min por execução (dados abortados após
isso devem ser marcados como NA). A avaliação empírica deve cobrir:
1. Profiling de Subrotinas e Modelagem de Complexidade Empírica: O código
deve ser instrumentado. Em vez de apenas reportar o tempo total, tente ajustar os
tempos coletados (em função do número de nós 𝑁 ) a curvas teóricas de complexidade
espacial e temporal usando ferramentas de regressão (ex: SciPy, Scikit-Learn). Relate
métricas de ajuste (como 𝑅2 ou RMSE). Discuta se gargalos de memória (ex: cópias
desnecessárias de matrizes, garbage collection) distorcem a complexidade teórica.
2. Impacto do Warm-Start na Árvore de Busca: Colete e compare o número de nós
expandidos e o número de nós podados na árvore do Branch-and-Bound para cada
uma das cinco estratégias de inicialização heurística citadas acima. Avalie o trade-off : o
tempo gasto na heurística inicial compensa a restrição do espaço de busca?
2
3. Análise de Sensibilidade baseada em características das instâncias (Instâncias
Adversariais): Extraia métricas geométricas/topológicas de cada instância de teste e
avalie como elas impactam o gap de aproximação (erro em relação ao ótimo). Considere
extrair features como (podem ser incluídas outras):
• Razão de aspecto (aspect ratio) do bounding box dos pontos.
• Proporção de pontos pertencentes ao fecho convexo.
• Entropia da distribuição espacial ou densidade de clustering.
• Estatísticas da Árvore Geradora Mínima: como o grau máximo de um nó
(presença de hubs).
• Coeficiente de variação das distâncias ao vizinho mais próximo (heterogeneidade
local).
Conjunto de Teste
Como conjunto de teste, serão utilizadas instâncias compiladas na biblioteca TSPLIB cuja
função de custo seja a distância euclidiana em 2D (tipo EUC_2D).
Links Úteis de Acesso
• Repositório Principal da TSPLIB: https://comopt.ifi.uni-heidelberg.de/software
/TSPLIB95/
• Instâncias: https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/tspindex.h
tml
Seleção e Limite de Tamanho das Instâncias
Para viabilizar a execução experimental e permitir a modelagem de regressão sem inviabilizar
o tempo de computação dos estudantes, o conjunto de teste deve focar na faixa de 𝑁 = 10
a 𝑁 = 75 vértices (selecionando entre 12 e 16 instâncias, como ulysses16, gr24, fri26,
bays29, swiss42, att48):
1. Faixa Principal (𝑁 ≤ 40): A grande maioria das instâncias (80% do conjunto escolhido)
deve situar-se nesta faixa. O Branch-and-Bound deve conseguir resolver a maioria
dessas instâncias dentro do prazo de 30 minutos (especialmente com warm-start),
garantindo a coleta de estatísticas ricas sobre contagem de nós, profundidade e tempo
por subrotina.
3
2. Faixa Limite (𝑁 ∈ [45, 75]): Instâncias maiores atuam como “testes de estresse”. Para
a busca exata (Branch-and-Bound), algumas dessas instâncias deverão estourar o tempo
limite de 30 minutos (sendo registradas como NA, mas mantendo outros dados como
número de nós expandidos e podados até a interrupção), enquanto os algoritmos aproximados
(Twice-around-the-tree e Christofides) continuarão finalizando em segundos. Isso
permitirá analisar exatamente onde ocorre o ponto de colapso do algoritmo exato e o
ganho real das heurísticas em problemas de maior escala.
Relatório (Artigo Científico)
Deverá ser escrito um relatório em formato de artigo científico (padrão SBC em LaTeX) contendo:
1. Introdução: Apresentação do problema e do trabalho.
2. Implementação, Profiling e Auditoria: Descreva as estruturas de dados adotadas.
Se ferramentas de IA generativa foram usadas para code-generation, detalhe as refatorações
realizadas por você para evitar estouro de memória, escolha da estratégia da
árvore (best/depth-first), limitações do código bruto e como o profiling ajudou a isolar
as subrotinas mais caras (como o Matching no Christofides).
3. Experimentos e Resultados (Modelagem):
• Apresente os modelos de regressão do comportamento assintótico (curvas ajustadas).
• Discuta a eficiência da poda na árvore gerada através das estratégias de warm-start.
• Avalie o peso das features no desempenho/qualidade da solução (Análise de Sensibilidade).
Construa um modelo (ex: árvore de decisão ou regressão múltipla)
para correlacionar quais características da instância (fecho convexo, clusters, etc.)
pioram ou facilitam o desempenho heurístico/aproximado.
4. Conclusões: Resumo das descobertas empíricas.
O que entregar?
Devem ser entregues todos os arquivos fonte, a base de dados com as métricas coletadas
(tempos, memória, features das instâncias, nós podados), e os relatórios (PDF e código-fonte
LaTeX). Se utilizar LLMs, um anexo com os históricos de prompts utilizados também é obrigatório.
Entrega via repositório no GitHub.
4
Política de Avaliação e Uso de Inteligência Artificial
• O uso de IAs generativas (ex: ChatGPT, Copilot) como assistentes de codificação é permitido
e encorajado. Contudo, o critério de avaliação desloca-se da “escrita sintática”
para o domínio da engenharia e da teoria. Códigos entregues sem refatoração crítica
geralmente apresentarão gargalos severos de eficiência.
• Arguição: O professor e monitor se reservam no direito de arguir os alunos sobre decisões
algorítmicas de baixo nível submetidas. A nota estará condicionada à capacidade
do grupo de defender tecnicamente o comportamento estrutural e empírico do código
(ex: explicar por que uma estratégia de limite inferior falhou em uma instância adversarial
específica), provando o ganho pedagógico. Trabalhos copiados (de colegas ou não
compreendidos feitos por IA) sem essa robustez auditada terão nota anulada.
Cronograma
• Entrega final: 24/11/2026 às 23h59.
• Atraso: não serão aceitas entregas fora do prazo
Lembrem-se que os experimentos podem levar dias de processamento. É imprescindível iniciar
a implementação o mais breve possível.

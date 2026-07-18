EEL891 - 2026.01 - Trabalho 2
Regressão: avaliar imóveis a partir de suas características


EEL891 - 2026.01 - Trabalho 2

Submit Prediction
Overview
Start

17 days ago
Close

6 days to go
Description
Este é o segundo trabalho de avaliação da disciplina EEL891 (Introdução ao Aprendizado de Máquina) para a turma do semestre 2025.02.

Neste trabalho você utilizará técnicas de regressão multivariável para estimar o preço de um imóvel a partir de características tais como o tipo de imóvel (apartamento, casa, loft ou quitinete), bairro onde está localizado, número de quartos, número de vagas, área útil, área extra e presença de elementos diferenciais em relação a outros imóveis, tais como churrasqueira, estacionamento para visitantes, piscina, playground, quadra esportiva, campo de futebol, salão de festas, salão de jogos, sala de ginástica, sauna e vista para o mar.

Instruções
Instruções
Baixe os arquivos de dados do projeto, disponíveis na aba "Data", onde você encontrará também a descrição dos arquivos e o significado de todos os campos neles contidos.
Implemente o pré-processamento adequado (tratamento de atributos categóricos, utilização de fatores de escala, etc.).
Visualize, analise e selecione os atributos com maior potencial preditivo.
Escolha os modelos preditivos de sua preferência e construa um script Python para testá-los.
Verifique o desempenho dos modelos preditivos escolhidos por meio de validação cruzada, ou outra técnica de validação de sua preferência, usando o conjunto de treinamento fornecido.
Ajuste os hiperparâmetros e o conjunto de variáveis utilizadas pelo modelo preditivo, de modo a reduzir o erro obtido na validação cruzada.
Quando estiver satisfeito com o ajuste, treine o seu modelo preditivo usando o conjunto de treinamento completo, obtenha a previsão do modelo para o conjunto de teste fornecido e submeta ao Kaggle (que detém o gabarito oculto do conjunto de teste), para verificar o desempenho real do seu modelo preditivo.
Revise e repita os passos anteriores quantas vezes achar necessário, até ficar satisfeito com o resultado.
Escreva um relatório descrevendo todos os passos do seu trabalho (pré-processamento dos dados, seleção dos atributos, modelos preditivos experimentados e seus respectivos ajustes de hiperparâmetros, técnicas de validação utilizadas, resultados intermediários alcançados, etc.).
Envie um e-mail para o professor ( heraldo@poli.ufrj.br ) contendo:
o seu id no Kaggle
o código-fonte utilizado (para possibilitar a reprodução dos resultados relatados)
o seu relatório (o relatório poderá ser entregue em um arquivo separado no formato PDF, ou integrado no próprio código-fonte, caso seja utilizado Jupyter Notebook, Kaggle Notebook ou Google Colab).
Avaliação
Métrica de Erro Utilizada
As respostas submetidas ao Kaggle serão avaliadas e comparadas em termos da raiz quadrada do erro percentual quadrático médio, métrica conhecida como RMSPE, dada pela fórmula:

Composição da Nota do Trabalho
O professor atribuirá ao trabalho de cada aluno uma nota na faixa de 0 a 10.

A nota do trabalho terá a seguinte composição:

Código fonte: até 5 pontos
Relatório: até 2 pontos
Desempenho: até 3 pontos
Critério de Pontuação no Quesito Desempenho
O desempenho do modelo de regressão construído pelo aluno será pontuado na faixa de 0 a 3 pontos, de acordo com a posição alcançada no ranking da competição, da seguinte forma:

onde:

n é o número de alunos que participaram da competição
p é a posição obtida pelo aluno no ranking da competição
ou seja, o primeiro colocado no ranking ganhará a pontuação máxima de 3 pontos e as posições seguintes terão pontuações decrescentes até o último colocado, que receberá pontuação 3/n.

Prazo
Prazo
O horário de encerramento da competição e prazo limite para entrega do trabalho (código-fonte + relatório) será:

Sábado 18/JUL/2026 23:59 - horário local Rio de Janeiro (GMT - 3:00)
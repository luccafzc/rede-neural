# Metodologia da versão revisada

Adaptação para português do artigo [Classifying human vs. AI text with machine learning and explainable transformer models](https://pmc.ncbi.nlm.nih.gov/articles/PMC12686477/). Não é uma reprodução integral: redes recorrentes, embeddings estáticos e pruning estão fora desta versão.

1. Baixar PT-L2Detect na revisão `69e5e82a5c0e2edd0b7765e04cb3a2c902962d8e` e verificar SHA-256. Usar rótulos 0 humano / 1 IA.
2. Normalizar Unicode e espaços, preservar acentos/pontuação e retirar marcações de transcrição corretamente.
3. Remover duplicatas por ID/hash e pares quase idênticos por similaridade de caracteres ≥0,95. Preservar primeiro teste, depois validação e treino. O vetorizador da auditoria não entra no classificador.
4. Comparar seis pipelines em validação cruzada estratificada de três folds do treino, semente 42; vocabulários aprendidos dentro de cada fold. Critério: F1 macro médio. Hiperparâmetros dos candidatos definidos antes da avaliação.
5. Ajustar a família escolhida nos 1.385 textos de treino. Dividir a validação em dois grupos estratificados de 166: calibração sigmoide e compatibilidade por classe. Conservar os 701 textos de teste para diagnóstico histórico.
6. Gerar saída seletiva: somente uma classe compatível e condições de suporte atendidas permitem decisão. Documentar mínimos de comprimento, cobertura lexical, repetição e comprimento máximo do treino.
7. Reportar métricas binárias a 0,5, qualidade probabilística, cobertura total e por classe, número de inconclusivos, erros e métricas condicionais das decisões. Conservar manifestos de IDs e previsões sem redistribuir textos.
8. A comparação v1/v2 reutiliza o teste histórico. Toda conclusão de generalização exige base externa independente. Não interpretar p-valores de compatibilidade como probabilidade de autoria.

O relatório `REVISAO.md` apresenta achados, limitações e resultados. As duas frases da conversa foram usadas apenas como testes de comportamento: não entraram no treino, na seleção, na calibração nem na escolha do nível nominal.

O módulo transformer usa a mesma limpeza/deduplicação, 3 épocas, taxa 3e-5, lote efetivo 32 em uma GPU e 256 tokens. Mantém subconjuntos separados para checkpoint, temperatura e limiar e registra truncamentos. Foi revisado e testado em suas funções numéricas, mas não foi treinado nesta entrega. A política seletiva da versão CPU não foi aplicada a essa extensão.

# Revisão técnica e científica

## Conclusão

A versão inicial podia produzir uma classificação categórica para textos sem suporte experimental. A base restrita a aprendizes de português é o maior limite: melhorar arquitetura não elimina a diferença entre esse público e textos de falantes nativos. A v2 melhora a forma de representar texto e de reconhecer incerteza, mas ainda não sustenta uma alegação de detecção geral de autoria.

## Achados corrigidos

1. **Decisão forçada em frases curtas.** Não há exemplos de IA com menos de 50 palavras na base. O programa agora se abstém abaixo desse comprimento, independentemente da origem presumida da frase. Também controla o comprimento máximo estudado, cobertura lexical e repetição.
2. **Erro de limpeza dentro de palavras.** `ci<u>dade` era convertido em `ci dade`. Tags inline agora são removidas sem inserir espaços; separadores de parágrafo continuam recebendo espaços. Há teste de regressão e todo o modelo v2 foi retreinado com a correção.
3. **Representação limitada a vocabulário de palavras.** Foram comparadas seis configurações. A SVM que combina TF-IDF de palavras/bigramas e caracteres de 3–5 posições foi selecionada pela média de F1 macro de três folds do treino. Partes de palavras reduzem dependência de palavras completas conhecidas, sem garantir robustez a mudança de domínio.
4. **Pontuação sem calibração.** A SVM agora recebe calibração sigmoide em 166 exemplos não usados para ajustar o classificador. Outros 166 exemplos formam a referência de compatibilidade por classe. A documentação do scikit-learn exige separar os exemplos de treino dos de calibração; a implementação usa um estimador congelado. [Documentação](https://scikit-learn.org/stable/modules/calibration.html).
5. **Duplicatas aproximadas.** A auditoria lexical por cosseno de caracteres encontrou 31 pares com similaridade ≥0,95, incluindo 5 entre treino e validação. Foram retiradas 31 linhas adicionais do treino; o conjunto passou a 1.385 textos. O algoritmo usa componentes conexos e preserva primeiro teste e validação. Não é um detector de paráfrases nem de autores.
6. **Risco de apresentar somente métricas dos acertos.** O relatório separa decisão binária diagnóstica, cobertura, inconclusivos, falsos positivos e métricas condicionais das decisões. Abstenções não contam como acertos.
7. **Uso no VS Code.** Caminhos padrão agora são resolvidos junto ao script, há leitura de `.txt` UTF-8, mensagens de erro claras e uma suíte de testes incluída no projeto.
8. **Extensão transformer.** Foram adicionadas validações de rótulos, probabilidades e temperatura, guarda para falta de classe no ajuste do limiar, a mesma correção de dados e uma auditoria de truncamento. A inferência carrega a temperatura do relatório; não assume silenciosamente temperatura 1. Treinamento e XAI dessa extensão ainda precisam ser executados em ambiente adequado.

## Abstenção e seus limites

A referência guarda, para cada classe, o erro de probabilidade `1 - p(classe verdadeira)`. Para cada texto novo, calcula-se um p-valor de compatibilidade com cada classe; apenas um conjunto unitário permite a decisão. Conjuntos vazios ou com as duas classes são inconclusivos. O nível nominal usado é 0,05, com correção finita por `n+1`.

Essa abordagem se inspira na predição conformal condicionada à classe. A interpretação formal pressupõe exemplos intercambiáveis com a referência, o que não foi estabelecido para textos nativos, novos geradores, autores repetidos ou este protocolo de fontes. Não significa 95% de confiança na autoria individual. A restrição adicional de suporte também exige reportar cobertura. [Referência metodológica](https://arxiv.org/abs/2107.07511).

## Resultado observado e comparação honesta

O teste histórico conserva os mesmos 701 IDs da v1. Acurácia binária: 96,29% → 96,58%; F1 macro: 0,9560 → 0,9594. Entre as discordâncias, a v1 acertou sozinha 7 textos e a v2 acertou sozinha 9; McNemar exato p=0,8036. Isso não demonstra superioridade estatística. A escolha de modelo foi feita no treino, mas o conjunto histórico já foi observado durante desenvolvimento; não é uma nova avaliação cega.

Na operação seletiva, foram 181 humanos classificados como humanos, nenhum humano como IA, 5 textos de IA como humanos, 300 textos de IA como IA e 215 inconclusivos. A cobertura é 69,33%; entre humanos é 90,05% e entre IA é 61,00%. O resultado “zero falsos positivos” é uma observação desta amostra, não garantia futura. A calibração da v2 teve Brier 0,0329 e log-loss 0,1270 no diagnóstico; não é garantia de calibração fora desse domínio.

## O que ainda limita a assertividade

- Os rótulos humanos estão ligados a corpora de aprendizes e os rótulos de IA a outra origem. O modelo pode estar reconhecendo fonte, proficiência ou estilo.
- Todos os tópicos humanos estão ausentes. A separação temática completa não pode ser comprovada.
- A deduplicação não estabelece independência entre autores nem remove todas as paráfrases.
- A base é pequena e os textos de IA são mais regulares em comprimento. Não há cobertura de todas as formas de redação.
- Os seis candidatos e a extensão transformer não foram comparados em um novo conjunto de português nativo.

## Melhor sequência para avançar

1. **Dados de uso real antes de novos modelos.** Reunir parágrafos e redações em português nativo, com autoria documentada. Usar vários gêneros, temas, níveis de escolaridade e comprimentos. Para IA, registrar gerador, versão, prompt e parâmetros. Não fabricar rótulos a partir de avaliações de detectores.
2. **Pares comparáveis e grupos separados.** Para cada tema e faixa de tamanho, obter humanos e IA. Manter textos do mesmo autor, prompt ou documento de origem no mesmo grupo. Reservar conjuntos por tempo e por gerador não visto.
3. **Congelar um teste externo.** Separar treino, escolha do modelo, calibração e teste por grupos. O teste externo deve ser usado uma vez após as decisões. Medir falsos positivos humanos, recall de IA, F1 macro, calibração, cobertura e erros por subgrupo.
4. **Comparar transformer somente com esse protocolo.** Treinar BERTimbau e XLM-RoBERTa e comparar com a SVM revisada. Para textos maiores, comparar uma estratégia de janelas ou um modelo de contexto maior em vez de supor que os primeiros 256 tokens representam a redação inteira.
5. **Escolher o compromisso de uso.** Mais decisões podem trazer mais erros. Ajustar a política em validação, registrar o custo de inconclusivos e não alterar limites para acertar exemplos particulares do usuário.

Essa ordem é preferível a simplesmente aumentar um limiar de IA ou trocar o modelo sem mudar a base. Mudança de domínio é um desafio documentado em detecção de texto gerado. [MAGE, artigo de pesquisa](https://arxiv.org/abs/2305.13242).

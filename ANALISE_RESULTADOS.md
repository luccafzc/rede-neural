# Análise com tendência exploratória

A interface local (`interface.py`) e o backend Vercel (`app.py`) agora chamam a mesma função `analyze_bundle`. O HTML, o layout e o modelo treinado não foram modificados nesta etapa. A publicação no Vercel continua pendente.

## O problema identificado

A referência de compatibilidade do modelo salvo aceita a classe humana aproximadamente até score IA 0,238 e a classe IA aproximadamente a partir de 0,913. Uma grande região intermediária não tem classe compatível, embora o classificador produza uma pontuação. O filtro lexical, por sua vez, considera palavras completas e pode interromper a análise mesmo quando as características de partes de palavras da SVM híbrida estão disponíveis. Uma contagem superior a 535 palavras também impedia qualquer resultado.

Isso explica parte das abstenções. Não prova que as previsões descartadas estejam corretas ou que o modelo reconheça autoria em português nativo.

## Comportamento implementado

- A classificação seletiva permanece nos campos `prediction`, `label`, `score_ai`, `reasons` e compatibilidade. O terminal e a avaliação externa conservam esse protocolo.
- A resposta da análise acrescenta `estimate`, `analysis_warnings` e `analysis_version`. O campo `estimate` contém a pontuação real da SVM calibrada, sem alterar os pesos do modelo.
- Quando há uma classificação seletiva, a interface a apresenta. Caso contrário, apresenta **Tendência humana** para score abaixo de 0,5 ou **Tendência de IA** para score acima de 0,5, com o aviso visível **Estimativa exploratória**. Empate numérico é exibido como **Padrões equilibrados**, sem atribuição arbitrária de classe. A cor de incerteza é preservada nas estimativas.
- Vocabulário pouco semelhante à base gera um aviso, em vez de impedir a estimativa. Isso não significa que palavras desconhecidas foram aprendidas: o vetor de características disponível é usado pelo mesmo modelo.
- Textos com mais de 535 palavras são divididos em janelas contíguas de tamanhos equilibrados, cada uma de até 535 palavras. Todas as palavras entram uma única vez; não há truncamento nem repetição para preencher janelas. A tendência do documento usa a média dos scores ponderada pela contagem de palavras. O retorno completo inclui o score de cada trecho e avisa se eles divergem.
- A média de janelas **não é uma probabilidade calibrada de autoria do documento**. As janelas são operacionais e não necessariamente coincidem com frases. O resultado não identifica quais partes seriam humanas/IA nem comprova autoria mista.
- Textos abaixo de 50 palavras, excessivamente repetitivos, sem palavras ou com alguma janela sem características reconhecidas continuam sem estimativa. A tela apresenta uma orientação sobre a entrada. Não há garantia de reconhecer todo texto sem sentido, idioma diferente ou conteúdo fora do domínio.

A mudança é na política de apresentação/análise, não no treinamento. O classificador continua limitado à base PT-L2Detect. Nenhuma consulta é usada automaticamente para aprender, e o modelo não foi retreinado. Melhorar a taxa de acertos em textos variados requer dados rotulados representativos e teste independente.

## Medição realizada

Reexecução nos **701 textos do teste histórico**, obtidos da revisão original com SHA-256 verificado. Os hashes do teste foram conferidos com o manifesto local. O conjunto já foi usado no desenvolvimento; não é uma nova validação externa e não representa todos os textos em português.

| Saída | Decisões ou tendências exibidas | Sem resposta utilizável | Erros nas respostas exibidas |
| --- | ---: | ---: | ---: |
| Somente decisão seletiva original | 486 | 215 | 5 |
| Decisão seletiva + tendência exploratória | 694 | 7 | 24 |

Foram **208 tendências adicionais**, das quais **189 coincidiram com o rótulo e 19 erraram**. Os sete casos sem estimativa tinham menos de 50 palavras. A cobertura de saída sobe de **69,33% para 99,00%**. A acurácia entre as respostas exibidas passa de **98,97% para 96,54%**, em conjuntos de tamanhos diferentes. Portanto, houve mais respostas e mais erros, **sem evidência de aumento de acurácia**. Das 694 respostas, houve um humano apontado como IA e 23 textos de IA apontados como humanos.

As 486 decisões seletivas e suas métricas coincidiram com o relatório original. A análise por janelas foi exercitada no único texto histórico acima do limite; isso é insuficiente para validar generalização para artigos longos.

Arquivos: `resultados_analise/relatorio_analise.json` e `resultados_analise/predicoes_analise.json`. As previsões contêm hashes, rótulos e scores; não redistribuem os textos brutos.

Para reproduzir (o script baixa a base original se necessário):

```powershell
python scripts/avaliar_analise.py --data-dir .\data --output-dir .\resultados_analise
python -m unittest discover -s tests -v
```

O JSON registra as versões de execução. Este ambiente forneceu scikit-learn 1.8.0 e emitiu aviso de versão ao carregar o artefato salvo com 1.9.1; as métricas seletivas históricas foram reproduzidas. A compatibilidade de serialização entre versões não é garantida. A execução publicada deve usar um ambiente compatível com o artefato e ser verificada antes da apresentação.

## Para a apresentação

Usar textos com pelo menos 50 palavras e autoria conhecida permite comparar a saída com uma referência verificável. Textos encontrados no Google não têm, por isso apenas, garantia de autoria humana. Apresentar “tendência” como estimativa do classificador, incluindo os casos em que ele erra. O relatório acima permite explicar o compromisso entre quantidade de respostas e confiabilidade sem atribuir ao modelo uma capacidade não demonstrada.

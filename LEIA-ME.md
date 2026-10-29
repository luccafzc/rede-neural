# Detector acadêmico de texto — versão 2

Para o diagnóstico dos resultados inconclusivos, o roteiro de treinamento com textos do domínio desejado e a avaliação da base disponibilizada pelos autores do artigo, consulte [PLANO_TREINAMENTO.md](PLANO_TREINAMENTO.md).

O código foi revisado para analisar parágrafos e redações, com saída **Humano**, **IA** ou **Inconclusivo**. A versão atual combina palavras e trechos de palavras com SVM, calibração sigmoide e referência de compatibilidade por classe. O modelo já está treinado em `resultados_v2/modelo_selecionado.joblib`.

## Interface para apresentação

Na pasta do projeto, dê dois cliques em **Iniciar_Interface.bat**. A interface abre no navegador e usa o modelo v2 já treinado. Se preferir iniciar pelo terminal do VS Code:

```powershell
.\.venv\Scripts\python.exe interface.py
```

O endereço padrão é `http://127.0.0.1:8765`. Mantenha o terminal aberto durante a apresentação; pressione **Ctrl+C** para encerrar. Se a porta estiver ocupada, execute com `--port 8766`. A interface usa as dependências já instaladas do projeto, sem novas bibliotecas e sem retreinar.

Cole um parágrafo ou importe um arquivo `.txt` em UTF-8 e clique em **Analisar texto**. A tela mostra a classificação sustentada pela referência ou, quando há sinal utilizável mas a regra seletiva se abstém, **Tendência humana / Tendência de IA**, identificada como estimativa exploratória. Vocabulário diferente passa a ser uma ressalva da estimativa. Textos longos são divididos em trechos e recebem uma média ponderada das pontuações; essa média não é probabilidade de autoria. Textos com menos de 50 palavras, repetição excessiva ou trechos sem padrões reconhecidos recebem orientação para ajustar a entrada. **Explorar detalhes técnicos** abre o score, a cobertura de vocabulário e a resposta JSON. O score não representa certeza de autoria. Veja os resultados e limites em [ANALISE_RESULTADOS.md](ANALISE_RESULTADOS.md).

O processamento é local, sem envio a serviços externos, e o aplicativo não salva os textos. A interface destina-se à demonstração neste computador, sem hospedagem pública. Para uma instalação nova, siga os dois primeiros comandos da seção abaixo antes de abrir o atalho.

## Usar no VS Code

A pasta original deste projeto já foi atualizada. Se você usa uma cópia em outro local, extraia o pacote V2 em uma pasta nova. Use Python 3.12. Dentro da pasta que contém `detector_pt.py`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe detector_pt.py predict --text "Cole aqui seu parágrafo completo."
```

Se o ambiente `.venv` já está instalado, execute apenas o último comando. Para redações, é mais conveniente salvar o texto em um arquivo UTF-8:

```powershell
.\.venv\Scripts\python.exe detector_pt.py predict --text-file .\minha_redacao.txt
```

O código encontra o modelo padrão a partir da localização do próprio script. Caminhos explícitos fornecidos pelo usuário continuam relativos ao terminal atual. Não use o modelo antigo de `resultados` com `--model` se pretende testar a v2.

## Entender a resposta

- `prediction`: Humano, IA ou Inconclusivo.
- `label`: 0, 1 ou `null` para inconclusivo.
- `score_ai`: probabilidade calibrada na referência L2, quando calculada. Não é certeza de autoria. Fora das condições operacionais, é `null`.
- `compatible_labels`: classes compatíveis com a referência estatística. Uma decisão exige exatamente uma classe compatível.
- `compatibility_pvalues`: p-valores de compatibilidade; não são probabilidades de autoria.
- `reasons`: motivo da abstenção.
- `n_words` e `vocabulary_coverage`: características do texto informado.

O modelo fornecido opera entre **50 e 535 palavras**. Também pode se abster por baixa cobertura de vocabulário, repetição excessiva ou ambiguidade. Esses critérios não garantem que um texto aceito esteja no domínio: a base continua sendo português de aprendizes. Não aumente artificialmente o texto por repetição para ultrapassar o limite.

Esse é o comportamento seletivo preservado no comando `predict` e nas métricas históricas. As interfaces local e Vercel usam `analyze_bundle`: a resposta mantém esses campos e acrescenta `estimate` e `analysis_warnings`. Uma tendência não transforma `label: null` em uma decisão validada. A interface apresenta a tendência separadamente e não aprende com as consultas.

As duas frases testadas na conversa agora recebem Inconclusivo. Isso evita decidir sobre frases sem suporte na base; **não significa que o modelo descobriu a autoria delas**.

## O que foi medido

No diagnóstico histórico de 701 textos, a v2 com decisão binária forçada teve 96,58% de acurácia, contra 96,29% da regressão logística v1. A diferença é pequena e não significativa no teste pareado de McNemar (p=0,804); não há evidência de melhoria geral robusta.

Na saída operacional v2, **486 receberam decisão e 215 ficaram inconclusivos**. Foram **481 acertos e 5 erros entre as 486 decisões**, com cobertura de 69,33%. Não houve humanos classificados como IA nesse teste, mas houve 5 textos de IA classificados como humanos. A acurácia condicional de 98,97% refere-se apenas aos casos decididos, não aos 701 textos e não a qualquer texto novo.

O principal avanço comprovado é controle de incerteza e de condições de uso. Ainda falta uma avaliação externa independente com textos nativos. Os dados históricos já foram examinados durante o desenvolvimento.

## Refazer o experimento

```powershell
.\.venv\Scripts\python.exe detector_pt.py train
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

O primeiro comando baixa os dados, corrige duplicatas, compara seis configurações por validação cruzada no treino, calibra e grava `resultados_v2`. Os 19 testes verificam inferência, abstenção, limpeza, separação dos dados, métricas, calibração, explicações, execução por arquivo e integração da interface local com o modelo.

O notebook `Experimento_Humano_IA_Portugues.ipynb` foi atualizado e contém os três módulos completos. Faça upload no Colab e execute em ordem. Ele foi validado estruturalmente e quanto à sintaxe; não foi executado integralmente no Colab.

## Próxima avaliação: textos externos

Prepare um CSV com `text,label`, usando 0 para humano e 1 para IA, e origem conhecida. Não use outro detector para criar os rótulos. O arquivo `modelo_coleta.csv` contém apenas os cabeçalhos, sem exemplos artificiais de treinamento.

```powershell
.\.venv\Scripts\python.exe detector_pt.py evaluate --csv .\base_externa.csv
```

A avaliação não retreina o modelo. Rejeita duplicatas exatas e textos presentes no experimento anterior; autores, prompts e paráfrases também devem ser auditados. Leia `REVISAO.md` para os achados e a prioridade das próximas melhorias.

## Arquivos e status

- `detector_pt.py`: limpeza, utilidades, interface de comandos e experimento clássico de referência.
- `modelo_v2.py`: seleção, calibração, abstenção, explicação e avaliação externa.
- `interface.py`, `interface_web` e `Iniciar_Interface.bat`: interface local de apresentação e atalho de abertura.
- `transformer_pt.py`: extensão revisada com BERTimbau/XLM-RoBERTa, **ainda não treinada**.
- `resultados_v2`: resultados atuais, modelo, manifestos e auditoria.
- `resultados`: resultados históricos da v1, preservados para rastreabilidade; não representam a execução atual.

Os transformers continuam opcionais e usam as dependências de `requirements-transformers.txt`. A inferência dessa extensão limita a 256 tokens e registra truncamentos na avaliação. Ela ainda não tem a mesma política operacional de abstenção da versão CPU. LIME/SHAP também não foram executados.

Os textos brutos não são redistribuídos. O código baixa a revisão documentada da [PT-L2Detect](https://github.com/nxthang/ProAID---Proficiency-Calibrated-AI-Detection/tree/69e5e82a5c0e2edd0b7765e04cb3a2c902962d8e/PT-L2Detect%20Dataset), cujos termos variam por fonte e incluem uso acadêmico. Cite o artigo-base e os responsáveis pelos dados.

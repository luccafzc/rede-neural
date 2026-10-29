# Diagnóstico e plano de treinamento

Atualização operacional: a interface agora oferece estimativas exploratórias e análise por trechos. Consulte [ANALISE_RESULTADOS.md](ANALISE_RESULTADOS.md) para as medições. Isso não substitui o treinamento com textos variados em português descrito neste plano.

Avaliação em 29/10/2026. A interface foi refinada; o modelo e seus limiares não foram retreinados nesta revisão. Não há evidência de ganho de acurácia decorrente das mudanças visuais.

## Por que ocorre “Inconclusivo”

O classificador atual é uma SVM com TF-IDF de palavras e caracteres, calibração sigmoide e uma regra de compatibilidade por classe. Não é uma rede neural em execução: a extensão transformer existe, mas ainda não foi treinada. Colar textos na interface não atualiza o modelo; a aplicação não salva esses textos nem aprende com cada consulta.

O treino contém 1.385 redações de aprendizes de português (PT-L2Detect). Artigos científicos, notícias e textos de falantes nativos representam outros estilos e vocabulários. Um texto pode ter mais de 50 palavras e ainda estar fora do domínio de treinamento. Estar no Google também não comprova autoria humana; é necessário documentar a procedência.

As regras operacionais verificadas no código e no modelo são:

- Entre 50 e 535 palavras após limpeza.
- Cobertura mínima de aproximadamente 74% das palavras distintas elegíveis no vocabulário de referência.
- Proporção de palavras distintas de pelo menos 15% entre os tokens elegíveis.
- Exatamente uma classe com p-valor de compatibilidade superior a 0,05.

São filtros de suporte, não garantias de autoria. Alterá-los somente para obter mais respostas categóricas pode elevar os erros.

Nos resultados históricos (`resultados_v2/predicoes_teste.csv`), dos 701 textos, 215 (30,67%) receberam abstenção:

| Motivo | Textos |
| --- | ---: |
| Nenhuma classe compatível | 194 |
| Apenas baixa cobertura lexical | 13 |
| Apenas comprimento abaixo do mínimo | 6 |
| Comprimento curto e baixa cobertura | 1 |
| Comprimento acima do máximo | 1 |

Esses números descrevem o teste histórico, não os artigos inseridos pelo usuário. Sem esses artigos e sua origem documentada, não é possível quantificar os erros relatados. Os 98,97% de acurácia já documentados dizem respeito apenas às 486 decisões; não incluem as abstenções e não medem desempenho geral em artigos.

## Como treinar para o uso pretendido

1. **Definir a unidade de análise.** Começar por trechos coerentes em português, com tamanho registrado. Para artigos completos, coletar documentos longos e validar uma estratégia de janelas e agregação por documento. Recortar um trecho não permite concluir a autoria de todo o artigo; não aumentar o limite atual sem avaliação.
2. **Coletar dados comparáveis de ambas as classes.** Usar artigos, textos acadêmicos e notícias de autoria humana documentada, respeitando seus termos. Gerar exemplos de IA dos mesmos gêneros, assuntos e faixas de tamanho com diferentes modelos, versões e prompts. Evitar humanos só de uma fonte e IA só de outra. Textos mistos ou revisados por IA precisam de um protocolo próprio de rotulagem.
3. **Registrar a procedência.** Usar `modelo_coleta.csv`, com `text,label,group_id,source,author_id,generator,prompt_family,domain,language,created_at`. Para este projeto, 0 significa humano e 1 significa IA. Registrar também versão do gerador, parâmetros e permissão de uso em um manifesto. Não atribuir rótulos a partir de outro detector.
4. **Limpar e separar antes de ajustar.** Auditar duplicatas, paráfrases, textos vazios, codificação e rótulos conflitantes. Colocar todos os trechos do mesmo documento e textos relacionados por autor ou família de prompts no mesmo grupo. Separar treino, validação, calibração de probabilidade, calibração de compatibilidade e teste final por esses grupos. Ajustar vocabulário apenas no treino. O teste final deve ficar reservado até concluir as escolhas.
5. **Comparar modelos.** Começar pela SVM atual como referência; depois comparar BERTimbau ou XLM-RoBERTa usando as mesmas partições independentes. Quantidade de épocas e arquitetura não compensam uma base inadequada. O volume necessário deve ser avaliado por curvas de aprendizado; não existe número de exemplos que garanta eliminar os erros.
6. **Recalibrar no novo domínio.** Recriar a calibração sigmoide e a referência por classe usando exemplos separados do ajuste do classificador. Avaliar políticas candidatas em validação e escolher o compromisso entre quantidade de decisões e erros. Não escolher limiares pelo teste final nem por exemplos isolados.
7. **Medir o resultado completo.** Reportar falsos positivos em humanos, recall de IA, F1 macro, cobertura de decisões, abstenções e erros entre as decisões. Separar resultados por gênero, tamanho, fonte e gerador; avaliar também um gerador não visto no treino. Usar intervalos de incerteza, respeitando os grupos.

### O que já funciona no projeto

Para medir o modelo atual em uma base externa rotulada, sem retreinar:

```powershell
.\.venv\Scripts\python.exe detector_pt.py evaluate --csv .\base_externa.csv --output-dir .\avaliacao_externa
```

O comando rejeita duplicatas exatas e textos conhecidos do experimento anterior. A independência entre autores, documentos, prompts e paráfrases precisa ser auditada à parte.

**Atenção ao comando de treino:** `detector_pt.py train` recarrega a PT-L2Detect e refaz o experimento atual. Não aceita `--csv` e não incorpora automaticamente os textos da interface ou do arquivo de coleta. `--data-dir` é o diretório dos arquivos esperados dessa base, não um importador genérico. Para o novo corpus, será necessário implementar um carregador com validação de esquema e divisões por grupos, conectar essas partições à seleção/calibração e salvar o novo modelo em outro diretório. Repetir o comando atual com os mesmos dados não corrige a mudança de domínio.

## Avaliação do artigo indicado

Artigo: [PMC12686477](https://pmc.ncbi.nlm.nih.gov/articles/PMC12686477/). O texto integral não pôde ser consultado devido ao bloqueio de rede do ambiente. A seção “Data availability” fornecida pelo usuário afirma que as bases estão em Kaggle/GitHub, remete à Tabela 1 e informa ausência de licenças explícitas/data cards nas bases originais. A Tabela 1 não foi inspecionada; portanto, não foi confirmada a lista completa de fontes originais.

Foi possível inspecionar diretamente o [repositório dos autores](https://github.com/shamylafirdoos/Gpt-vs-Human-Text-Classification), revisão `82bf5888efba65bc9aed64f6b02f198d493f806d`:

- Contém README, licença MIT, notebook e o arquivo [shuffled_data.csv](https://github.com/shamylafirdoos/Gpt-vs-Human-Text-Classification/blob/82bf5888efba65bc9aed64f6b02f198d493f806d/shuffled_data.csv).
- O CSV tem **20.000 linhas**, colunas `Data` e `Labels`, com **10.000 rótulos 0 e 10.000 rótulos 1**. O significado desses códigos deve ser confirmado na documentação das fontes antes da conversão para o esquema local.
- Os exemplos inspecionados estão em inglês; não foi feita identificação automática do idioma de todas as linhas. O notebook usa BERT uncased, RoBERTa e XLM-RoBERTa e lê `/kaggle/input/gpt-vs-human-dataset/shuffled_data.csv`. Esse caminho não identifica o proprietário da base Kaggle.
- Não há colunas para autores, fontes, prompts, geradores, licenças ou grupos. Isso dificulta comprovar independência entre treino e teste.
- Auditoria do CSV inteiro: zero textos vazios, **143 registros excedentes por repetição** e **31 grupos de textos com rótulos conflitantes**, após normalizar espaços com `split/join` e caixa com `casefold`. Um grupo conflitante é um mesmo texto normalizado associado a mais de um rótulo. Essas contagens não incluem auditoria de paráfrases.
- Há sinais de problemas de codificação nos exemplos inspecionados. A limpeza e a procedência precisam ser revisadas antes de treinar.
- SHA-256 do CSV: `f9ab6ff46e38f4930fd3375e6ca5a56134c064855c029e5ed2ad08b3fa9e5c6b`.

**Conclusão:** sim, existe uma base para download no repositório indicado pelo artigo. É útil para estudar/reproduzir o experimento e avaliar métodos. Não está demonstrado que seja adequada para melhorar a detecção em português; misturá-la diretamente à PT-L2Detect ou traduzi-la automaticamente pode introduzir sinais de idioma/tradução em vez de autoria. A licença MIT do repositório não esclarece por si só os direitos das fontes originais. Consultar a Tabela 1 e os termos dessas fontes antes de incorporar/redistribuir dados.

O CSV foi inspecionado temporariamente e não foi incorporado ao treino nem redistribuído neste projeto.

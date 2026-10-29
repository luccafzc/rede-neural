'use strict';
const $ = id => document.getElementById(id);
let config = null, running = false, revision = 0, request = null;
const input = $('text-input');
function error(message='') { $('error').textContent = message; $('error').hidden = !message; }
function words(text) { return text.trim() ? text.trim().split(/\s+/u).length : 0; }
function clearResult() {
  revision++;
  if (request) request.abort();
  request = null; running = false;
  $('result-content').hidden = true; $('loading-state').hidden = true; $('empty-state').hidden = false;
  $('technical').open = false; $('result-panel').setAttribute('aria-busy','false');
  $('button-label').textContent = 'Analisar texto';
}
function updateCount() {
  const count = words(input.value), hint = $('count-hint');
  $('word-count').textContent = count.toLocaleString('pt-BR');
  $('analyze').disabled = !config || running || !input.value.trim();
  hint.classList.remove('outside');
  if (!config) return;
  const min=config.policy.min_words, max=config.policy.max_words;
  if (!count) hint.textContent='A contagem acompanha o texto enquanto você escreve.';
  else if (count<min) { hint.textContent=`Texto curto: abaixo de ${min} palavras, o resultado será inconclusivo.`; hint.classList.add('outside'); }
  else if (count>max) { hint.textContent=`Acima das ${max} palavras estudadas. O modelo retornará inconclusivo.`; hint.classList.add('outside'); }
  else hint.textContent='Comprimento dentro do intervalo estudado. Isso não garante uma decisão.';
}
function onEdit() { clearResult(); error(); $('file-name').textContent=''; updateCount(); }
input.addEventListener('input',onEdit);
$('clear').addEventListener('click',()=>{input.value='';$('file-input').value='';onEdit();input.focus();});
$('import-button').addEventListener('click',()=>$('file-input').click());
$('file-input').addEventListener('change',async(event)=>{
  const file=event.target.files[0]; if(!file)return;
  clearResult();error();updateCount();
  const ticket=revision;
  try{
    if(!file.name.toLowerCase().endsWith('.txt'))throw new Error('Selecione um arquivo .txt em UTF-8.');
    if(file.size>400000)throw new Error('O arquivo é grande demais. Use um texto de até 100.000 caracteres.');
    const text=new TextDecoder('utf-8',{fatal:true}).decode(await file.arrayBuffer());
    if(ticket!==revision)return;
    if(text.length>100000)throw new Error('O texto excede 100.000 caracteres.');
    input.value=text; $('file-name').textContent=file.name; updateCount();
  }catch(exc){if(ticket===revision)error(exc instanceof TypeError?'Não foi possível ler o arquivo. Salve-o em UTF-8 e tente novamente.':exc.message);}
  finally{event.target.value='';}
});
function showResult(result){
  const known=result.label===0||result.label===1;
  $('outcome').className='outcome '+(result.label===0?'human':result.label===1?'ai':'uncertain');
  $('result-kicker').textContent=known?'CLASSIFICAÇÃO DO MODELO':'SEM CONCLUSÃO SUFICIENTE';
  $('prediction').textContent=result.label===0?'Classificado como humano':result.label===1?'Classificado como IA':'Inconclusivo';
  $('explanation').textContent=known?
    (result.label===0?'O texto foi compatível com a classe humana na referência utilizada pelo modelo.':'O texto foi compatível com a classe de IA na referência utilizada pelo modelo.'):
    'O modelo não tem suporte suficiente para atribuir uma única classificação a este texto.';
  $('reasons').replaceChildren();
  for(const reason of result.reasons||[]){const li=document.createElement('li');li.textContent=reason;$('reasons').append(li);}
  $('reasons').hidden=!(result.reasons||[]).length;
  $('analyzed-count').textContent=`${result.n_words} palavras`;
  $('score').textContent=result.score_ai===null?'Não calculada':Number(result.score_ai).toLocaleString('pt-BR',{minimumFractionDigits:4,maximumFractionDigits:4});
  $('coverage').textContent=Number(result.vocabulary_coverage).toLocaleString('pt-BR',{style:'percent',maximumFractionDigits:1});
  $('compatible').textContent=result.compatible_labels?(result.compatible_labels.join(' e ')||'Nenhuma'):'Não avaliadas';
  $('raw-result').textContent=JSON.stringify(result,null,2);
  $('empty-state').hidden=true;$('loading-state').hidden=true;$('result-content').hidden=false;
}
async function analyze(){
  if(!config||running||!input.value.trim())return;
  error();clearResult();running=true;updateCount();
  const ticket=revision;request=new AbortController();
  $('empty-state').hidden=true;$('loading-state').hidden=false;$('result-panel').setAttribute('aria-busy','true');
  $('button-label').textContent='Analisando…';
  try{
    const response=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json','X-Local-UI':'1'},body:JSON.stringify({text:input.value}),signal:request.signal});
    const result=await response.json();
    if(ticket!==revision)return;
    if(!response.ok)throw new Error(result.error||'Não foi possível analisar o texto.');
    showResult(result);
  }catch(exc){
    if(ticket===revision&&exc.name!=='AbortError'){
      $('loading-state').hidden=true;$('empty-state').hidden=false;
      error(exc instanceof TypeError?'A interface perdeu a conexão. Verifique se o terminal continua aberto.':exc.message);
    }
  }finally{
    if(ticket===revision){running=false;request=null;$('button-label').textContent='Analisar texto';$('result-panel').setAttribute('aria-busy','false');updateCount();}
  }
}
$('analyze').addEventListener('click',analyze);
input.addEventListener('keydown',event=>{if((event.ctrlKey||event.metaKey)&&event.key==='Enter'){event.preventDefault();analyze();}});
async function initialize(){
  try{
    const response=await fetch('/api/config');if(!response.ok)throw new Error();config=await response.json();
    $('range-hint').textContent=`Cole de ${config.policy.min_words} a ${config.policy.max_words} palavras. Textos fora do intervalo ficam inconclusivos.`;
    const names={hibrido_svm:'SVM híbrida',hibrido_logistica:'Regressão híbrida',palavras_svm:'SVM de palavras',palavras_logistica:'Regressão de palavras',caracteres_logistica:'Regressão de caracteres',palavras_random_forest:'Random Forest'};
    $('model-name').textContent=names[config.model]||config.model;
    updateCount();
  }catch{
    error('Não foi possível carregar o modelo. Inicie a interface pelo arquivo Iniciar_Interface.bat.');
  }
}
initialize();

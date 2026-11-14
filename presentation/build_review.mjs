// Editorial academic review. Author through artifactSession.run(build).
import fs from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
const ROOT=fileURLToPath(new URL('..',import.meta.url)).replaceAll('\\','/');
const B={paper:'#F6F3EC',ink:'#152A35',muted:'#566570',accent:'#B84E2E',line:'#D9D8D1'};
export async function build(p,session){
 const calibration=JSON.parse(await fs.readFile(ROOT+'/reports/kaggle-calibration/calibration-evaluation.json','utf8'));
 const benchmark=JSON.parse(await fs.readFile(ROOT+'/reports/policy-comparison.json','utf8'));
 function text(s,body,x,y,w,h,size=28,color=B.ink,font='Arial',bold=false){const o=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});o.text=body;o.text.style={typeface:font,fontSize:size,color,bold,insets:0,autoFit:'none',verticalAlignment:'top'};return o;}
 function slide(title,notes,dark=false){const s=p.slides.add({width:1280,height:720});s.background.fill=dark?B.ink:B.paper;s.speakerNotes.text=notes;const c=dark?B.paper:B.ink;if(title)text(s,title,70,55,1130,110,46,c,'Georgia');text(s,String(p.slides.items.length).padStart(2,'0'),1150,665,70,25,16,dark?'#AFC2CB':B.muted);return s;}
 function paragraph(s,head,body,x,y,w){text(s,head,x,y,w,40,28,B.accent,'Arial',true);text(s,body,x,y+54,w,125,27);}
 function table(s,values,x,y,w,widths,rowHeight=64){const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:rowHeight*values.length,columnWidths:widths,values});t.borders.assign({style:'solid',fill:B.line,width:.7});for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){const a=t.getCell(r,c);a.fill=r===0?B.ink:B.paper;a.text.style={typeface:'Arial',fontSize:24,color:r===0?B.paper:B.ink,bold:r===0,insets:12,verticalAlignment:'middle'};}return t;}
 async function image(s,file,x,y,w,h){const b=await fs.readFile(ROOT+'/'+file);s.images.add({dataUrl:'data:image/png;base64,'+b.toString('base64'),fit:'contain',alt:'Actual TrustRAG application',position:{left:x,top:y,width:w,height:h}});}
 const ref='Sources: Ding et al., Rowen, https://arxiv.org/abs/2402.10612 ; Sun et al., ReDeEP, https://arxiv.org/abs/2410.11414 ; Farquhar, Kossen, Kuhn and Gal, Nature 630 (2024), https://www.nature.com/articles/s41586-024-07421-0';
 {
 const s=slide('', 'Present TrustRAG as a completed, locally executable research prototype. Its controlled evaluation gives limited evidence and does not establish broad hallucination reduction.',true);
 text(s,'TrustRAG',70,125,1120,140,112,B.paper,'Georgia');
 text(s,'Uncertainty-Gated Retrieval\nand Hallucination Analysis',77,315,1060,140,42,'#C5D3D9');
 text(s,'Project review\n15 November 2026',77,570,900,70,22,'#C5D3D9');
 }
 {
 const s=slide('A plausible answer can still be unsupported', 'This is a recorded answer from the November 13 no-retrieval baseline, not an invented quote. The fictional Yarrow handbook says 08:00. Source: reports/policy-comparison.json, first record.');
 text(s,'“The Yarrow laboratory is open from\n9:00 AM to 5:00 PM on Saturdays\nand Sundays.”',72,220,1100,220,43,B.ink,'Georgia');
 text(s,'The supplied handbook says it opens at 08:00.',72,515,1110,65,29,B.accent,'Arial',true);
 text(s,'Recorded output without retrieval. Controlled fictional example.',72,610,1100,35,19,B.muted);
 }
 {
 const s=slide('Project objective',ref+' The goal is an inspectable pipeline, not a guarantee that an LLM will always be correct.');
 text(s,'Estimate uncertainty.\nFind relevant evidence.\nInspect the answer’s support.',72,205,1100,275,52,B.ink,'Georgia');
 text(s,'A working system that records why it answered, revised, or abstained.',72,555,1100,80,29,B.accent);
 }
 {
 const s=slide('The implemented pipeline','Adaptive mode generates an initial answer, samples meanings, and either exits unverified or retrieves. The always-retrieve mode now skips the unused initial answer. Context arrives before internal diagnosis. Source: backend/pipeline.py.');
 const steps=[['01','Generate and sample','Initial answer and stochastic variations'],['02','Group meanings','NLI equivalence and discrete entropy'],['03','Retrieve and inspect','Local evidence, contextual answer, ECS and PKS'],['04','Verify or abstain','Guarded correction and sentence-level support']];
 steps.forEach(([n,h,b],i)=>{const y=190+i*109;text(s,n,72,y,95,65,42,B.accent,'Georgia');text(s,h,195,y,940,43,30,B.ink,'Arial',true);text(s,b,195,y+46,940,40,25,B.muted);});
 }
 {
 const s=slide('A local research workspace','Actual screenshot of the reviewed app. The saved research section reports historical controlled experiments. Document lookup is extractive. Model mode uses cached local weights. Source: webapp and reports/review-workspace.png.');
 await image(s,'reports/review-workspace.png',64,176,850,455);
 text(s,'Ask a question\n\nInspect sources\n\nCompare routes\n\nExport the record',954,210,260,355,27);
 }
 {
 const s=slide('Uncertainty comes from meaning disagreement',ref+' Recorded adaptive Yarrow example: three samples formed three separate NLI clusters, giving ln(3)=1.099 nats. Extra claims and flawed NLI decisions can affect grouping. Low entropy is not a truth guarantee. Source: reports/policy-comparison.json.');
 text(s,'H = − Σ p(c) ln p(c)',72,205,1130,85,54,B.accent,'Georgia');
 table(s,[['Recorded sample','Meaning'],['1','Seasonal opening dates'],['2','10:00 AM to 4:00 PM'],['3','9:00 AM to 5:00 PM']],72,345,750,[300,450],59);
 text(s,'1.099',870,355,335,115,76,B.ink,'Georgia');text(s,'nats\n3 samples, 3 clusters',875,475,320,100,25,B.muted);
 }
 {
 const s=slide('Calibration uses separate source groups','The 64 claims come from fictional handbooks, two opposing claims per source. Training uses 16 source groups, validation 8, test 8. The code rejects group overlap and identical contexts across splits. Source: data/calibration_cases.json; scripts/calibrate.py.');
 text(s,'64',72,225,390,170,125,B.accent,'Georgia');text(s,'controlled claims',80,410,390,55,31);
 table(s,[['Split','Claims','Source groups'],['Training','32','16'],['Validation','16','8'],['Test','16','8']],530,210,675,[275,175,225],80);
 text(s,'Training selects components. Validation selects the threshold. Test measures the result.',72,590,1130,70,25,B.muted);
 }
 {
 const s=slide('The fitted context-consistency detector','The backbone language model is frozen. Train-time search selects from four candidate layers and four heads. A regularized, standardized logistic regression fits ECS/PKS features. Validation chooses a threshold of about 0.960. This is a project adaptation inspired by ReDeEP, not the paper component-selection reproduction. Source: artifacts/calibration.json and scripts/calibrate.py.',true);
 text(s,'Layer 26',72,230,570,110,78,B.paper,'Georgia');text(s,'Head 01',710,230,500,110,78,B.paper,'Georgia');
 text(s,'External Context Score\nAttention and hidden-state similarity',76,415,550,115,28,'#C5D3D9');
 text(s,'Parametric Knowledge Score\nChange across the FFN residual stream',714,415,490,120,28,'#C5D3D9');
 text(s,'The small detector is trained. The language model weights stay frozen.',76,590,1110,50,25,'#C5D3D9');
 }
 {
 const s=slide('Detection on the controlled test split','Test labels: hallucinated=1. TP=3, FP=1, TN=7, FN=5. Sixteen claims from eight unseen source groups. AUROC=0.734375. F1=0.5. Accuracy=.625. These are small controlled-data results and do not establish general reliability. Source: reports/kaggle-calibration/calibration-evaluation.json.');
 text(s,'0.734',72,200,520,150,100,B.accent,'Georgia');text(s,'Test AUROC',80,355,500,55,31);
 table(s,[['Actual / prediction','Flagged','Not flagged'],['Contradictory','3','5'],['Faithful','1','7']],610,205,595,[295,145,155],86);
 text(s,'F1  0.50     Precision  0.75     Recall  0.375',72,535,1125,60,30);
 text(s,'The detector misses five of eight contradictory claims.',72,610,1125,40,24,B.muted);
 }
 {
 const s=slide('Retrieval helped on eight unseen handbooks','Historical November 13 GPU experiment. The task is opening-time extraction plus saved NLI support. Extra supported closing times are allowed. Eight test handbooks per policy, separate from calibration train/validation sources. No general benchmark claim. Source: reports/policy-comparison.json.');
 table(s,[['Policy','Opening-time check','Mean elapsed'],['No retrieval','0 / 8','1.11 s'],['Always retrieve','8 / 8','1.46 s'],['Adaptive, correction off','8 / 8','3.14 s'],['Adaptive, guarded correction','8 / 8','3.12 s']],72,205,1135,[550,300,285],76);
 text(s,'All retrieval variants passed this narrow extraction check.',72,610,1130,40,25,B.accent);
 }
 {
 const s=slide('Adaptive routing added overhead here','Native editable chart from the historical November 13 policy comparison. Values are mean latency seconds. All historical modes included a context-free initial generation. The November 15 release removes that redundant generation from always-retrieve, so the chart is not a timing benchmark of the current release. Adaptive sampling generated more tokens: 1116 total versus 332 for always-retrieve. Source: reports/policy-comparison.json.');
 const chart=s.charts.add('bar',{position:{left:72,top:200,width:800,height:390},categories:['Always retrieve','Adaptive, off','Adaptive, guarded'],series:[{name:'Seconds',values:[1.4558575,3.1381525,3.1150825],fill:B.accent}],barOptions:{direction:'column',gapWidth:100},hasLegend:false,chartFill:B.paper,plotAreaFill:B.paper,xAxis:{textStyle:{typeface:'Arial',fontSize:20,fill:B.ink}},yAxis:{textStyle:{typeface:'Arial',fontSize:20,fill:B.muted},numberFormatCode:'0.0',majorGridlines:{fill:B.line,width:1}},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.00',textStyle:{typeface:'Arial',fontSize:23,fill:B.ink}}});
 text(s,'No observed\naccuracy gain',928,230,275,150,36,B.ink,'Georgia');text(s,'No intervention triggered\nin these eight questions.',928,425,275,115,25,B.muted);
 text(s,'Historical GPU run. Latest release removes an unused generation from always-retrieve.',72,620,1135,35,19,B.muted);
 }
 {
 const s=slide('Evidence checks before and after correction','The current implementation checks numeric agreement before NLI, keeps common title abbreviations intact during claim splitting, and requires support for every claim. Intervention only runs for an unsupported flagged candidate. A rejected revision cannot replace the original answer. These checks are conservative, not proof of truth. Source: backend/evidence.py and backend/models.py.');
 paragraph(s,'First, reject unsupported numbers','09:30 cannot match a source that only says 09:00.',72,205,1080);
 paragraph(s,'Then, check every sentence','Each claim needs numeric agreement and an NLI entailment match.',72,380,1080);
 text(s,'Guarded intervention keeps only revisions that pass the evidence check.',72,585,1120,65,28,B.accent);
 }
 {
 const s=slide('Review improvements in the final release','Source changes on November 15: always-retrieve goes directly to retrieval/contextual generation, numeric support guard, abbreviation-aware splitting, calibration shape/finite-value validation, and correct EOS token accounting. Regression suite and real-weight smoke checks are detailed in the repository.');
 table(s,[['Review finding','Implemented change'],['Unused initial answer','One generation call for always-retrieve'],['Numbers could slip through NLI','Numeric agreement before entailment'],['Fragile sentence boundaries','Abbreviation-aware claim splitting'],['Malformed fitted profiles','Schema and finite-value checks']],72,215,1135,[470,665],76);
 }
 {
 const s=slide('What the results support','Avoid claiming that all hallucinations are eliminated. The data is fictional and only one relation/question family. The detector has modest recall. NLI can fail, numerical token inclusion does not capture every relation contradiction, and low entropy can accompany a false answer. The intervention effectiveness is not established by a run with no interventions.');
 paragraph(s,'Completed implementation','A runnable application with local inference, fitted probes, evidence checks, and reproducible experiments.',72,210,1100);
 paragraph(s,'Limited research evidence','A small controlled test supports retrieval. It leaves general reliability and correction effectiveness unresolved.',72,410,1100);
 }
 {
 const s=slide('Repository and demonstration','The repository contains reviewed source, tests, fitted detector, experiment outputs, and this presentation. The model weights download through the documented explicit setup command. No personal user corpus, credentials, or local model caches are published.');
 text(s,'github.com/Black-and-Yellow/\nadditional_project',72,205,1130,145,42,B.ink,'Georgia');
 text(s,'1. Ask a question and inspect the retrieved passage\n2. Compare routes and detector measurements\n3. Export the full execution record',76,415,1110,170,28);
 }
 {
 const s=slide('Research foundations',ref+' Model: HuggingFaceTB/SmolLM2-135M-Instruct, pinned revision in backend/models.py. NLI: cross-encoder/nli-MiniLM2-L6-H768. Project adaptation and controlled results should be distinguished from the source papers.');
 text(s,'Rowen',72,200,320,55,34,B.accent,'Georgia');text(s,'Adaptive retrieval through consistency\nDing et al.  ·  SIGIR-AP 2025',430,200,760,95,27);
 text(s,'Semantic entropy',72,350,320,110,34,B.accent,'Georgia');text(s,'Uncertainty over answer meanings\nFarquhar et al.  ·  Nature 2024',430,350,760,95,27);
 text(s,'ReDeEP',72,505,320,65,34,B.accent,'Georgia');text(s,'Context and parametric knowledge diagnosis\nSun et al.  ·  ICLR 2025',430,505,760,95,27);
 }
 const dir=ROOT+'/presentation/.review';await fs.mkdir(dir,{recursive:true});
 for(const [i,s]of p.slides.items.entries()){const png=await s.export({format:'png',scale:1});await fs.writeFile(`${dir}/slide-${String(i+1).padStart(2,'0')}.png`,new Uint8Array(await png.arrayBuffer()));}
 console.log('Authored and rendered '+p.slides.items.length+' slides.');
}

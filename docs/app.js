(() => {
  const $ = (s, ctx=document) => ctx.querySelector(s);
  const staticMode = new URLSearchParams(location.search).has('static');
  const $$ = (s, ctx=document) => [...ctx.querySelectorAll(s)];

  // Scroll progress
  const progress = $('#scrollProgress');
  const updateProgress = () => {
    const h = document.documentElement.scrollHeight - innerHeight;
    progress.style.width = `${h ? (scrollY / h) * 100 : 0}%`;
  };
  addEventListener('scroll', updateProgress, {passive:true}); updateProgress();

  // Reveal on scroll
  const io = new IntersectionObserver(entries => entries.forEach(e => {
    if(e.isIntersecting){e.target.classList.add('in'); io.unobserve(e.target)}
  }), {threshold:.12});
  $$('.reveal').forEach(el => io.observe(el));

  // 3D tilt cards
  const prefersReduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  if(!prefersReduced){
    $$('.tilt').forEach(card => {
      card.addEventListener('pointermove', e => {
        const r = card.getBoundingClientRect();
        const x = (e.clientX-r.left)/r.width - .5;
        const y = (e.clientY-r.top)/r.height - .5;
        const depth = Number(card.dataset.depth || 8);
        card.style.transform = `perspective(900px) rotateX(${(-y*depth).toFixed(2)}deg) rotateY(${(x*depth).toFixed(2)}deg) translateY(-2px)`;
      });
      card.addEventListener('pointerleave', () => card.style.transform='');
    });
  }

  // Interactive pipeline details
  const details = [
    ['01','Curated thyroid evidence','Guidelines and educational references are the source of truth. Every downstream step preserves document, page, section, and chunk provenance.',['Traceability','Controlled corpus','Page metadata']],
    ['02','Focused token-aware chunks','The benchmarked production configuration uses 200-token chunks with zero overlap to reduce context noise and improve passage focus.',['200 tokens','0 overlap','1,470 chunks']],
    ['03','BGE-small embeddings','BAAI/bge-small-en-v1.5 was selected after measured embedding comparisons because it improved early ranking and MRR while staying practical.',['384 dimensions','Normalized','CPU practical']],
    ['04','Frozen Top-4 retrieval','ChromaDB returns the four strongest candidate evidence chunks. The retrieval window was selected by measured Precision@K and Hit@K trade-offs.',['ChromaDB','Top-K 4','No reranker']],
    ['05','Deterministic evidence IDs','Retrieved passages are packaged as [E1]...[E4]. The LLM cites these IDs while deterministic code owns the real document/page/chunk metadata.',['E1–E4','Provenance','No fake metadata']],
    ['06','Evidence-only generation','Groq openai/gpt-oss-120b generates structured JSON under instructions that the retrieved evidence is the only clinical source of truth.',['Temperature 0','Structured JSON','Evidence-only']],
    ['07','Claim-level verification','Generated claims are checked for evidence IDs, metadata resolution, clinical support, and numeric/unit consistency. Ambiguous cases can become REVIEW_REQUIRED.',['Claim support','Numeric checks','Citation resolution']],
    ['08','Final safety decision','The safety policy combines input risk, evidence quality, and post-generation validation to answer, answer with caution, abstain, or redirect.',['ALLOWED','NEEDS_CAUTION','REFUSE_REDIRECT']]
  ];
  const detail = $('#pipelineDetail');
  $$('.pipeline-node').forEach(btn => btn.addEventListener('click', () => {
    $$('.pipeline-node').forEach(b=>b.classList.remove('active')); btn.classList.add('active');
    const d = details[Number(btn.dataset.step)];
    detail.innerHTML = `<div class="detail-index">${d[0]}</div><div><strong>${d[1]}</strong><p>${d[2]}</p></div><div class="detail-tech">${d[3].map(x=>`<span>${x}</span>`).join('')}</div>`;
  }));

  // Showcase tabs
  const img = $('#showcaseImage'), cap = $('#showcaseCaption');
  $$('.showcase-tabs button').forEach(btn => btn.addEventListener('click', () => {
    $$('.showcase-tabs button').forEach(b=>b.classList.remove('active')); btn.classList.add('active');
    img.style.opacity='.15'; img.style.transform='scale(.985)';
    setTimeout(()=>{img.src=btn.dataset.img; cap.textContent=btn.dataset.title; img.onload=()=>{img.style.opacity='1';img.style.transform='';}},120);
  }));

  // Lightbox from module cards + showcase image
  const lightbox=$('#lightbox'), lbimg=$('.lightbox img');
  const openLightbox = src => {lbimg.src=src;lightbox.classList.add('open');lightbox.setAttribute('aria-hidden','false')};
  $$('.module-card').forEach(card=>card.addEventListener('click',()=>openLightbox(card.dataset.image)));
  img.addEventListener('click',()=>openLightbox(img.src));
  $('.lightbox-close').addEventListener('click',()=>lightbox.classList.remove('open'));
  lightbox.addEventListener('click',e=>{if(e.target===lightbox)lightbox.classList.remove('open')});
  addEventListener('keydown',e=>{if(e.key==='Escape')lightbox.classList.remove('open')});

  // Canvas pseudo-3D thyroid particle field
  const canvas=$('#thyroidCanvas'), ctx=canvas.getContext('2d');
  const dpr=Math.min(devicePixelRatio||1,2); const base=700;
  canvas.width=base*dpr; canvas.height=base*dpr; ctx.scale(dpr,dpr);
  const pts=[];
  function thyroidShape(x,y){
    // two lobes + isthmus; x/y normalized roughly -1..1
    const l=((x+.48)**2/.42**2 + (y+.03)**2/.72**2)<1;
    const r=((x-.48)**2/.42**2 + (y+.03)**2/.72**2)<1;
    const bridge=(Math.abs(x)<.48 && Math.abs(y)<.16 && (1-(Math.abs(x)/.5))>.25);
    const notch=(x*x/.20**2 + (y+.58)**2/.20**2)<1;
    return (l||r||bridge) && !notch;
  }
  for(let i=0;i<1050;i++){
    let x,y,tries=0; do{x=Math.random()*2.1-1.05;y=Math.random()*2-1;tries++}while(!thyroidShape(x,y)&&tries<100);
    const z=(Math.random()-.5)*.6; pts.push({x,y,z,p:Math.random()*Math.PI*2,s:.55+Math.random()*1.6});
  }
  let t=0, mx=0,my=0;
  canvas.addEventListener('pointermove',e=>{const r=canvas.getBoundingClientRect();mx=((e.clientX-r.left)/r.width-.5)*.8;my=((e.clientY-r.top)/r.height-.5)*.5});
  canvas.addEventListener('pointerleave',()=>{mx=0;my=0});
  function render(){
    t+=.006;ctx.clearRect(0,0,base,base);
    const grad=ctx.createRadialGradient(350,350,50,350,350,310);grad.addColorStop(0,'rgba(42,189,255,.10)');grad.addColorStop(.45,'rgba(40,139,255,.035)');grad.addColorStop(1,'rgba(0,0,0,0)');ctx.fillStyle=grad;ctx.fillRect(0,0,700,700);
    const ry=t+mx, rx=.18+my;
    const proj=pts.map(p=>{
      let x=p.x,y=p.y,z=p.z;
      let cx=Math.cos(ry),sx=Math.sin(ry); let x1=x*cx-z*sx,z1=x*sx+z*cx;
      let cy=Math.cos(rx),sy=Math.sin(rx); let y1=y*cy-z1*sy,z2=y*sy+z1*cy;
      const scale=1.65/(2.1-z2);return {x:350+x1*238*scale,y:350+y1*238*scale,z:z2,s:p.s,ph:p.p};
    }).sort((a,b)=>a.z-b.z);
    for(let i=0;i<proj.length;i++){
      const p=proj[i], a=.24+(p.z+.4)*.42 + Math.sin(t*3+p.ph)*.08;
      ctx.beginPath();ctx.arc(p.x,p.y,p.s*(1.05+p.z*.35),0,Math.PI*2);ctx.fillStyle=`rgba(${p.z>.05?'54,218,255':'46,132,255'},${Math.max(.08,Math.min(.9,a))})`;ctx.fill();
    }
    // sparse connections
    ctx.lineWidth=.6;
    for(let k=0;k<90;k++){
      const a=proj[(k*11)%proj.length], b=proj[(k*37+19)%proj.length]; const dx=a.x-b.x,dy=a.y-b.y,dist=Math.hypot(dx,dy);
      if(dist<72){ctx.strokeStyle='rgba(50,186,255,.11)';ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke()}
    }
    if(!staticMode) requestAnimationFrame(render);
  } render();
})();

from pathlib import Path
from copy import deepcopy
from zipfile import ZipFile
import re,json,hashlib
from lxml import etree
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.part import Part
from docx.opc.packuri import PackURI
from docx.opc.constants import RELATIONSHIP_TYPE as RT

P=Path(__file__).resolve().parents[1]
(P/'tmp/entrega').mkdir(parents=True,exist_ok=True)
config=json.loads((P/'config/entrega.json').read_text(encoding='utf-8'))
plantilla=P/'tmp/entrega/plantilla.docx'
repls={b'http://purl.oclc.org/ooxml/wordprocessingml/main':b'http://schemas.openxmlformats.org/wordprocessingml/2006/main',b'http://purl.oclc.org/ooxml/officeDocument/relationships':b'http://schemas.openxmlformats.org/officeDocument/2006/relationships',b'http://purl.oclc.org/ooxml/drawingml/':b'http://schemas.openxmlformats.org/drawingml/2006/'}
with ZipFile(P/'docs/conference-template-letter.docx') as origen,ZipFile(plantilla,'w') as copia:
 for info in origen.infolist():
  datos=origen.read(info.filename)
  if info.filename.endswith(('.xml','.rels')):
   for viejo,nuevo in repls.items():datos=datos.replace(viejo,nuevo)
  copia.writestr(info,datos)
# Normalizar longitudes Strict de la copia temporal, conservando geometría.
with ZipFile(plantilla) as z: partes={i.filename:(i,z.read(i.filename)) for i in z.infolist()}
with ZipFile(P/'tmp/entrega/plantilla_autoria.docx','w') as z:
 for nombre,(info,d) in partes.items():
  if nombre in ['word/document.xml','word/styles.xml']:
   d=re.sub(rb'="(-?[0-9.]+)pt"',lambda m:b'="'+str(round(float(m.group(1))*20)).encode()+b'"',d)
  z.writestr(info,d)
doc=Document(P/'tmp/entrega/plantilla_autoria.docx')
body=doc._element.body
for elemento in list(body):
 if elemento.tag!=qn('w:sectPr'):body.remove(elemento)
sec=doc.sections[0]
sec.page_width=Inches(8.5);sec.page_height=Inches(11)
sec.top_margin=Inches(.75);sec.bottom_margin=Inches(1)
sec.left_margin=Inches(.62014);sec.right_margin=Inches(.62014)
cols=sec._sectPr.find(qn('w:cols'));cols.set(qn('w:num'),'1');cols.set(qn('w:space'),'360')
for hijo in list(cols):cols.remove(hijo)
for nombre in ['Normal','Body Text','Abstract']:
 s=doc.styles[nombre];s.font.name='Times New Roman';s.font.size=Pt(10);s.font.color.rgb=RGBColor(0,0,0)
 s.paragraph_format.space_after=Pt(4);s.paragraph_format.line_spacing=1
for estilo_n in doc.styles:
 if estilo_n.type==1:
  for n in estilo_n.element.xpath('./w:pPr/w:numPr'):n.getparent().remove(n)
for nombre in ['Heading 1','Heading 2']:
 s=doc.styles[nombre];s.font.name='Times New Roman';s.font.size=Pt(10)
 s.paragraph_format.space_before=Pt(9);s.paragraph_format.space_after=Pt(5)
 for n in s.element.xpath('./w:pPr/w:numPr'):n.getparent().remove(n)
for nombre in ['references','figure caption']:
 doc.styles[nombre].font.name='Times New Roman';doc.styles[nombre].font.size=Pt(8)
for s in doc.sections:
 for p in s.header.paragraphs:p.text=''
 for p in s.footer.paragraphs:p.text=''

def parrafo(texto,estilo='Body Text'):
 p=doc.add_paragraph(style=estilo)
 p.paragraph_format.widow_control=True
 for i,parte in enumerate(re.split(r'(\*\*.*?\*\*)',texto)):
  r=p.add_run(parte[2:-2] if parte.startswith('**') else parte)
  r.bold=parte.startswith('**')
 return p

def enlace(parrafo,texto,url):
 ident=doc.part.relate_to(url,RT.HYPERLINK,is_external=True)
 h=OxmlElement('w:hyperlink');h.set(qn('r:id'),ident)
 r=OxmlElement('w:r');rp=OxmlElement('w:rPr');c=OxmlElement('w:color');c.set(qn('w:val'),'17365D');rp.append(c);r.append(rp)
 t=OxmlElement('w:t');t.text=texto;r.append(t);h.append(r);parrafo._p.append(h)

def figura(nombre):
 p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
 ancho={'confusion_cnn':3.25,'v2_informe_confusion':2.6}.get(nombre,3.48)
 shape=p.add_run().add_picture(str(P/'docs/figuras'/f'{nombre}.png'),width=Inches(ancho))
 p.paragraph_format.keep_with_next=True;p.paragraph_format.space_after=Pt(2)
 svg=Part(PackURI(f'/word/media/{nombre}.svg'),'image/svg+xml',(P/'docs/figuras'/f'{nombre}.svg').read_bytes(),doc.part.package)
 ident=doc.part.relate_to(svg,RT.IMAGE)
 blip=shape._inline.xpath('.//a:blip')[0];exts=etree.SubElement(blip,'{http://schemas.openxmlformats.org/drawingml/2006/main}extLst')
 ext=etree.SubElement(exts,'{http://schemas.openxmlformats.org/drawingml/2006/main}ext',uri='{96DAC541-7B7A-43D3-8B79-37D633B846F1}')
 x=etree.SubElement(ext,'{http://schemas.microsoft.com/office/drawing/2016/SVG/main}svgBlip');x.set(qn('r:embed'),ident)

def tabla(lineas):
 filas=[[x.strip() for x in l.strip().strip('|').split('|')] for l in lineas if not re.match(r'^\|[\s:|-]+\|$',l)]
 n=len(filas[0]);t=doc.add_table(rows=0,cols=n);t.autofit=False
 anchos=([.95]+[(3.5-.95)/(n-1)]*(n-1)) if n>2 else [1.3,2.2]
 for rownum,fila in enumerate(filas):
  cells=t.add_row().cells
  for j,txt in enumerate(fila):
   cells[j].width=Inches(anchos[j]);p=cells[j].paragraphs[0]
   p.paragraph_format.space_after=Pt(3);p.paragraph_format.space_before=Pt(2);p.paragraph_format.line_spacing=1
   r=p.add_run(txt);r.font.name='Times New Roman';r.font.size=Pt(8);r.bold=rownum==0
  pr=t.rows[-1]._tr.get_or_add_trPr();pr.append(OxmlElement('w:cantSplit'))
  if rownum==0:pr.append(OxmlElement('w:tblHeader'))
 borders=OxmlElement('w:tblBorders')
 for pos in ['top','bottom','insideH']:
  e=OxmlElement('w:'+pos);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'3');e.set(qn('w:color'),'A0A0A0');borders.append(e)
 t._tbl.tblPr.append(borders)
 doc.add_paragraph().paragraph_format.space_after=Pt(0)

texto=(P/'report.md').read_text(encoding='utf-8-sig')
texto=re.sub(r'^\*\*Repositorio:\*\*.*$', '**Repositorio:** '+config['repositorio'], texto, flags=re.MULTILINE)
texto=re.sub(r'^\*\*Video:\*\*.*$', '**Video:** '+(config['video'] or 'ENLACE_PENDIENTE'), texto, flags=re.MULTILINE)
(P/'report.md').write_text(texto,encoding='utf-8')
lineas=texto.splitlines();titulo=lineas[0][2:]
p=doc.add_paragraph(style='paper title');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=p.add_run(titulo);r.font.size=Pt(22);r.font.color.rgb=RGBColor(0,0,0);r.font.name='Times New Roman'
p=doc.add_paragraph(', '.join(config['integrantes']),style='Author');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
parrafo('Algoritmos y Programación III · Semestre 2026-2','Affiliation').alignment=WD_ALIGN_PARAGRAPH.CENTER
p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after=Pt(3)
enlace(p,'Repositorio del proyecto',config['repositorio'])
p.add_run('   |   ')
if config['video']:enlace(p,'Video de presentación',config['video'])
else:p.add_run('Video de presentación pendiente de enlace')
sec=doc.add_section(WD_SECTION.CONTINUOUS);cols=sec._sectPr.find(qn('w:cols'));cols.set(qn('w:num'),'2');cols.set(qn('w:space'),'360')
for hijo in list(cols):cols.remove(hijo)
# Omitir título/autoría/enlaces ya compuestos arriba.
inicio=next(i for i,l in enumerate(lineas) if l=='## Resumen')
i=inicio;bloque=[];enrefs=False
while i<len(lineas):
 l=lineas[i].strip()
 if not l:i+=1;continue
 if l.startswith('## '):
  h=l[3:];enrefs=h=='Referencias';doc.add_paragraph(h,style='Heading 1');i+=1;continue
 if l.startswith('|'):
  filas=[]
  while i<len(lineas) and lineas[i].strip().startswith('|'):filas.append(lineas[i].strip());i+=1
  tabla(filas);continue
 if l.startswith('!['):figura(re.match(r'!\[([^]]+)\]',l).group(1));i+=1;continue
 if l.startswith('**Figura'):
  parrafo(l.replace('**',''),'figure caption');i+=1;continue
 estilo='references' if enrefs else ('Abstract' if i<inicio+4 else 'Body Text')
 p=parrafo(l,estilo)
 if enrefs:p.paragraph_format.space_after=Pt(3)
 i+=1
# Metadatos sin autores ficticios de plantilla.
doc.core_properties.title=titulo;doc.core_properties.author='; '.join(config['integrantes']);doc.core_properties.subject='Proyecto final de Algoritmos y Programación III';doc.core_properties.comments=''
out=P/'docs/informe_final.docx';doc.save(out)
print('DOCX generado',out)
print('Enlace de video:',config['video'] or 'PENDIENTE, no presentar como PDF definitivo')
# Evidencia de plantilla inalterada.
a=P/'tmp/entrega/artifact.md';a.write_text((a.read_text(encoding='utf-8-sig') if a.exists() else '')+'\nSHA plantilla original: '+hashlib.sha256((P/'docs/conference-template-letter.docx').read_bytes()).hexdigest()+'\n',encoding='utf-8')

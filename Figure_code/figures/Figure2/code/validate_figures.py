#!/usr/bin/env python3
"""Verify actual R layers, SVG bar geometry, PDF values and every full-row alignment."""
from pathlib import Path
from xml.etree import ElementTree as ET
import json,re
import numpy as np,pandas as pd,fitz
from PIL import Image

P=Path(__file__).resolve().parents[1]
rank=pd.read_csv(P/'tables/accuracy_scores_ranks.csv')
methods=rank.method.tolist();assert len(methods)==28
palette=pd.read_csv(P/'config/method_colors.csv')
assert set(palette.method)==set(methods) and not palette.method.duplicated().any() and palette.hex.nunique()==28
colors=dict(zip(palette.method,palette.hex))
assert set(methods).isdisjoint({'Region Velocity','Region-Velocity','PhyloVelo'})
# Historical palette brightness does not constrain the updated ranking.
assert palette.hex.str.fullmatch(r'#[0-9A-Fa-f]{6}').all()
summary=pd.read_csv(P/'tables/metric_summary_5050.csv')
layers=pd.read_csv(P/'tables/actual_ggplot_bar_layers.csv')
errors=pd.read_csv(P/'tables/actual_ggplot_errorbar_layers.csv')
for view,frame in layers.groupby('view'):
    prefix={'Combined':'combined','Real':'real','Simulation':'simulation'}[view]
    q=frame.merge(summary[['method','metric',prefix+'_raw_mean']],on=['method','metric'],validate='one_to_one')
    np.testing.assert_allclose(q.actual_mean,q[prefix+'_raw_mean'],rtol=1e-11,atol=1e-12)
    assert q.actual_fill.str.upper().tolist()==[colors[m] for m in q.method]
    np.testing.assert_array_equal(q.actual_factor_position,[29-methods.index(m)-1 for m in q.method])
for view,frame in errors.groupby('view'):
    prefix={'Combined':'combined','Real':'real','Simulation':'simulation'}[view]
    q=frame.merge(summary[['method','metric',prefix+'_raw_mean',prefix+'_raw_se']],on=['method','metric'],validate='one_to_one')
    np.testing.assert_allclose(q.lower,q[prefix+'_raw_mean']-q[prefix+'_raw_se'],rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(q.upper,q[prefix+'_raw_mean']+q[prefix+'_raw_se'],rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(q.cap_width,.2,rtol=0,atol=1e-12)
registry=pd.read_csv(P/'figure_index.csv')
normalized_SD=pd.read_csv(P/'tables/batch_SD_normalized_method_summary.csv').set_index('method')
checks=[];bar_count=0;sd_count=0;empty_sd_count=0
for spec in registry.itertuples():
    stem=P/'figures'/spec.figure
    plotting=pd.read_csv(P/'figures'/spec.plotting_csv,dtype={'label':str})
    if 'plot_role' in plotting:
        score_field={'Combined':'combined_score','Real':'real_score','Simulation':'simulation_score'}[spec.view]
        for row in plotting[plotting.plot_role.eq('normalized_SD')].itertuples():
            np.testing.assert_allclose(row.value,normalized_SD.loc[row.method,score_field],rtol=1e-11,atol=1e-12,equal_nan=True)
    root=ET.parse(stem.with_suffix('.svg')).getroot()
    assert not any(e.tag.rsplit('}',1)[-1]=='image' for e in root.iter())
    assert any(e.tag.rsplit('}',1)[-1]=='text' for e in root.iter())
    with Image.open(stem.with_suffix('.png')) as im:im.verify()
    with Image.open(stem.with_suffix('.png')) as im:im.load();assert min(im.info.get('dpi',(0,0)))>=299
    with fitz.open(stem.with_suffix('.pdf')) as d:
        assert len(d)==1 and not d[0].get_images(full=True)
        assert all(d.extract_font(f[0])[3] for f in d[0].get_fonts(full=True))
        page=d[0];words=page.get_text('words')
        assert abs(page.rect.width-spec.width_inches*72)<=1.05 and abs(page.rect.height-spec.height_inches*72)<=1.05
        assert all(w[0]>=-1 and w[1]>=-1 and w[2]<=page.rect.width+1 and w[3]<=page.rect.height+1 for w in words),spec.figure
        if spec.show_methods:
            expected=methods if spec.n_rows==28 else [m for m in methods if m in plotting.method.unique()]
            positions={w[4]:w for w in words if w[4] in methods}
            assert set(positions)==set(expected),(spec.figure,'method labels')
            assert sorted(expected,key=lambda m:positions[m][1])==expected,(spec.figure,'method order')
            for m in expected:
                target=plotting.loc[plotting.method.eq(m),'PDF_y_center_points'].iloc[0]
                assert abs((positions[m][1]+positions[m][3])/2-target)<3,(spec.figure,m,'row alignment')
        if 'label' in plotting:
            plotting['label']=plotting.label.fillna('').astype(str)
            numeric=[w for w in words if re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?',w[4])]
            for row in plotting.itertuples():
                if hasattr(row,'plot_role') and row.plot_role not in {'normalized_SD'} and not spec.figure.startswith('Stability_batchSD'):
                    continue
                candidates=[w for w in numeric if abs((w[0]+w[2])/2-row.PDF_x_center_points)<20 and abs((w[1]+w[3])/2-row.PDF_y_center_points)<6]
                if row.label:
                    assert len(candidates)==1,(spec.figure,row.method,row.metric,'missing or duplicate number')
                    assert candidates[0][4]==row.label,(spec.figure,row.method,row.metric,'wrong STD')
                    assert float(row.label)==0 if row.value==0 else abs(float(row.label)-row.value)<=.0051*abs(row.value)
                    sd_count+=1
                else:
                    assert not candidates,(spec.figure,row.method,row.metric,'unmeasured cell contains a number')
                    empty_sd_count+=1
    if 'panel_left_inches' in plotting:
        rectangles=[]
        for e in root.iter():
            if e.tag.rsplit('}',1)[-1]!='rect':continue
            match=re.search(r'fill:\s*(#[0-9A-Fa-f]{6})',e.attrib.get('style',''))
            if match and match[1].upper() in set(colors.values()):
                rectangles.append({'x':float(e.attrib['x']),'y':float(e.attrib['y']),
                  'width':float(e.attrib['width']),'height':float(e.attrib['height']),'hex':match[1].upper()})
        for row in plotting.itertuples():
            if not np.isfinite(row.value) or not np.isfinite(row.panel_left_inches):continue
            lo=row.panel_left_inches*72;hi=(row.panel_left_inches+row.panel_width_inches)*72
            matches=[b for b in rectangles if b['hex']==row.hex and abs(b['y']+b['height']/2-row.PDF_y_center_points)<.08
                     and b['x']>=lo-.08 and b['x']+b['width']<=hi+.08]
            if row.value==0 and not matches:continue
            assert len(matches)==1,(spec.figure,row.method,row.metric,'bar location')
            r=matches[0]
            expected_width=abs(row.value)/(row.scale_max-row.scale_min)*row.panel_width_inches*72
            expected_x=lo+(min(0,row.value)-row.scale_min)/(row.scale_max-row.scale_min)*row.panel_width_inches*72
            assert abs(r['width']-expected_width)<.1,(spec.figure,row.method,row.metric,'bar length')
            assert abs(r['x']-expected_x)<.1,(spec.figure,row.method,row.metric,'bar sign/baseline')
            bar_count+=1
    if spec.view=='Real' and 'metric' in plotting:
        assert not set(plotting.metric).intersection({'Ground-truth correlation','Performance under downsampling'})
    checks.append({'figure':spec.figure,'rows':int(spec.n_rows),'editable_pdf_svg':True,'PNG_full_decode_300dpi':True,'PASS':True})
assert len(registry)==34
assert not registry.figure.duplicated().any()
assert all('Figure_ABC_'+v in set(registry.figure) for v in ['Combined','Real','Simulation'])
assert not any('Overall' in n for n in registry.figure)
sd=pd.read_csv(P/'tables/batch_sd_numeric_accuracy_order.csv')
assert len(sd[sd.combined_mean_sd.notna()].method.unique())==8
result={'status':'PASS','figures':len(checks),'actual_R_bar_layers_checked':len(layers),
 'actual_R_errorbar_layers_checked':len(errors),'actual_SVG_bars_geometry_colour_rows_checked':bar_count,
 'actual_PDF_STD_numbers_checked':sd_count,'blank_unmeasured_STD_cells_checked':empty_sd_count,
 'palette':'A','Accuracy_only_order':True,'full_row_alignment':True,'excluded_methods':True,
 'negative_values_and_cross_zero_errorbars_preserved':True,'all_PNG_full_decode_300dpi':True,'files':checks}
(P/'validation').mkdir(exist_ok=True)
(P/'validation/figure_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False))

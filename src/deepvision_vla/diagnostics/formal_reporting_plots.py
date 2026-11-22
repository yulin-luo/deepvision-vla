"""Scientific PNG/PDF reporting from already admitted aggregate records only.

The renderer never reads a model result, selects sources, changes layers,
smooths curves, imputes missing values or computes another model prediction.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

COLORS=('#0072B2','#D55E00','#009E73','#CC79A7')
MODEL_LABELS={'openvla':'OpenVLA','pi0':'Bridge π₀','qwen_h16':'QwenVLA-OFT'}
DISPLAY={'roi_matched_random':'Matched ROI','random_nonROI_visual':'Random non-ROI visual',
         'roi_matched':'Matched task ROI','non_target_object':'Non-target object','empty_background':'Background'}


def _alignment(fig,axes):
    fig.canvas.draw();width,height=fig.get_size_inches()*72
    boxes=[ax.get_position().bounds for ax in axes]
    if len(boxes)>1:
        if (max(b[2]*width for b in boxes)-min(b[2]*width for b in boxes)>1.5
                or max(b[3]*height for b in boxes)-min(b[3]*height for b in boxes)>1.5):
            raise ValueError('Rendered comparable plot areas are not equally sized within1.5pt')
        if len(boxes)==4:
            if (abs(boxes[0][0]-boxes[2][0])*width>1.5 or abs(boxes[1][0]-boxes[3][0])*width>1.5
                    or abs(boxes[0][1]-boxes[1][1])*height>1.5 or abs(boxes[2][1]-boxes[3][1])*height>1.5):
                raise ValueError('Rendered shared panel edges are not aligned within1.5pt')
    return {'status':'passed_equal_plot_area_sizes' if len(boxes)>1 else 'single_panel',
        'tolerance_pt':1.5,'figure_size_points':[float(width),float(height)],
        'axes_rectangles_points':[[float(b[0]*width),float(b[1]*height),float(b[2]*width),float(b[3]*height)] for b in boxes]}


def _values(stats,indices):
    import numpy as np
    return tuple(np.asarray([stats[key][i] if stats[key][i] is not None else np.nan for i in indices],dtype=float)
        for key in ('means','lower','upper'))


def _curve(ax,stats,indices,label,color):
    import numpy as np
    x=np.arange(11)/10;mean,lower,upper=_values(stats,indices)
    ax.plot(x,mean,label=label,color=color,marker='o',markersize=2.6,linewidth=1.4)
    if np.isfinite(lower).any():ax.fill_between(x,lower,upper,color=color,alpha=.13,linewidth=0)


def _style(ax,*,delta=False):
    ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=8)
    ax.set_xlim(-.02,1.02);ax.set_xticks([0,.2,.4,.6,.8,1]);ax.set_xlabel('Nominal retained-token fraction',fontsize=8)
    ax.grid(axis='y',color='#dddddd',linewidth=.5);ax.set_axisbelow(True)
    if delta:ax.axhline(0,color='#777777',linewidth=.7,linestyle='--')


def _note(panel):
    suffix='' if panel['status']=='complete_metric_panel' else ' • INCOMPLETE metric coverage'
    return f"Requested {panel['requested_sources']}; qualified {panel['qualified_sources']}; paired {panel['paired_sources']}{suffix}"


def _coverage(result,metric):
    counts=result.get('terminal_status_counts',{})
    return (f"Model totals: actual entries {result.get('actual_model_entries','unknown')}; "
        f"aliases {counts.get('within_run_alias',0)}; history {counts.get('history_reuse',0)}; "
        f"missing {counts.get('missing_terminal',0)}; unknown entry {result.get('unknown_model_entries','unknown')}; "
        f"metric-valued points {result.get('metric_reference_coverage_points',{}).get(metric,'unknown')}")


def _save(fig,axes,path,contract):
    import matplotlib.pyplot as plt
    if contract.get('synthetic_fixture_only'):
        fig.text(.5,.5,'SYNTHETIC FIXTURE',ha='center',va='center',rotation=25,fontsize=32,color='#777777',alpha=.12)
    alignment=_alignment(fig,axes)
    fig.savefig(path.with_suffix('.png'),dpi=220,facecolor='white')
    fig.savefig(path.with_suffix('.pdf'),facecolor='white')
    with path.with_suffix('.alignment.json').open('x',encoding='utf-8') as f:json.dump(alignment,f,indent=2)
    plt.close(fig);return {**contract,'filename':path.name,'alignment_status':alignment['status']}


def plot_report(report,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .formal_reporting import LAYERS,KEEPS
    out=Path(output);out.mkdir(exist_ok=False);contracts=[]
    synthetic=report.get('input_seals',{}).get('synthetic_fixture_only') is True
    with matplotlib.rc_context({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none',
            'axes.linewidth':.7,'legend.frameon':False}):
        for model,result in report['models'].items():
            model_label=MODEL_LABELS[model]
            for key,panel in result['panels'].items():
                group,metric=panel['group'],panel['metric']
                for statistic,suffix in [('absolute_error','absolute'),('signed_baseline_difference','delta')]:
                    stats=panel[statistic];delta=statistic=='signed_baseline_difference'
                    if group=='primary':
                        fig,ax=plt.subplots(figsize=(7.2,4.25));axes=[ax]
                        fig.subplots_adjust(left=.12,right=.98,bottom=.24,top=.8)
                        for i,layer in enumerate(LAYERS[model]):
                            indices=[panel['points'].index(f"{panel['regions'][0]}|L{layer}|K{k}") for k in range(11)]
                            _curve(ax,stats,indices,'Layer '+str(layer),COLORS[i])
                        _style(ax,delta=delta);ax.legend(ncol=4,loc='lower left',bbox_to_anchor=(0,1.02),fontsize=8)
                        ax.set_ylabel('Difference from baseline MSE' if delta else 'MSE',fontsize=9)
                    else:
                        fig,grid=plt.subplots(2,2,figsize=(7.6,6.1),sharex=True,sharey=True);axes=list(grid.flat)
                        fig.subplots_adjust(left=.11,right=.98,bottom=.23,top=.84,wspace=.27,hspace=.42)
                        for ax,layer in zip(axes,LAYERS[model]):
                            for i,region in enumerate(panel['regions']):
                                indices=[panel['points'].index(f'{region}|L{layer}|K{k}') for k in range(11)]
                                _curve(ax,stats,indices,DISPLAY[region],COLORS[i])
                            _style(ax,delta=delta);ax.set_title('Layer '+str(layer),fontsize=9,loc='left')
                        axes[0].legend(ncol=len(panel['regions']),loc='lower left',bbox_to_anchor=(0,1.28),fontsize=8)
                        axes[0].set_ylabel('Baseline difference MSE' if delta else 'MSE',fontsize=8)
                        axes[2].set_ylabel('Baseline difference MSE' if delta else 'MSE',fontsize=8)
                    title=f"{model_label} • {panel['label']}"
                    fig.suptitle(title,fontsize=10,y=.985)
                    fig.text(.11,.12,_note(panel),fontsize=7.5)
                    fig.text(.11,.082,_coverage(result,metric),fontsize=6.6)
                    fig.text(.11,.045,'Equal episode weights • pointwise 95% paired-source CI • aliases are not replicates',fontsize=7)
                    if model=='qwen_h16':fig.text(.11,.012,'Layer 0 masking runs before DeepStack reinjection.',fontsize=6.8)
                    if panel['paired_sources']==0:
                        for ax in axes:ax.text(.5,.5,'No complete paired metric sources',transform=ax.transAxes,ha='center',fontsize=8)
                    name=f'{model}__{metric}__{group}_{suffix}'
                    contracts.append(_save(fig,axes,out/name,{'model':model,'metric':metric,'group':group,'statistic':statistic,
                        'synthetic_fixture_only':synthetic,
                        'layers':list(LAYERS[model]),'keep_ratios':list(KEEPS),'paired_sources':panel['paired_sources'],
                        'status':panel['status'],'missing_imputed':False,'smoothing':False,'same_metric_space_only':True}))
            for comparison in result['shallow_deep']:
                group=comparison['group']
                panel=result['panels'][group+'|'+comparison['metric']]
                regions=panel['regions']
                fig,ax=plt.subplots(figsize=(7.2,4.25));fig.subplots_adjust(left=.12,right=.98,bottom=.24,top=.81)
                for i,region in enumerate(regions):
                    indices=[comparison['points'].index(f'{region}|K{k}') for k in range(11)]
                    _curve(ax,comparison['statistics'],indices,'Full primary ROI' if group=='primary' else DISPLAY[region],COLORS[i])
                _style(ax,delta=True);ax.legend(fontsize=8);ax.set_ylabel('Last-layer minus first-layer MSE',fontsize=9)
                fig.suptitle(f"{model_label} • {panel['label']} • Layer {comparison['last_layer']} − {comparison['first_layer']}",fontsize=9)
                fig.text(.12,.12,_note(panel)+' • fixed first-versus-last',fontsize=7.2)
                fig.text(.12,.082,_coverage(result,comparison['metric']),fontsize=6.6)
                fig.text(.12,.045,'Pointwise 95% paired-source CI • negative differences retained',fontsize=7)
                name=f"{model}__{comparison['metric']}__{group}_shallow_deep"
                contracts.append(_save(fig,[ax],out/name,{'model':model,'metric':comparison['metric'],'group':group,
                    'synthetic_fixture_only':synthetic,
                    'comparison':'last_selected_minus_first_selected','first_layer':comparison['first_layer'],'last_layer':comparison['last_layer'],
                    'keep_ratios':list(KEEPS),'negative_differences_allowed':True}))
    with (out/'PLOT_CONTRACTS.json').open('x',encoding='utf-8') as f:json.dump(contracts,f,indent=2)
    return contracts



def plot_primary_overview(report, output, report_sha256):
    """Draw the four registered layers for each native primary metric."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from .formal_plan import LAYERS, REPORT_METRICS

    units = {
        "openvla": "H1 action GT MSE",
        "pi0": "H4 dataset-reference MSE",
        "qwen_h16": "H16 prediction drift (native MSE)",
    }
    labels = {
        "primary_object_robot_proxy": "Historical object and robot ROI",
        "primary_task_roi": "Object, robot and interaction ROI",
    }
    models = ("openvla", "pi0", "qwen_h16")
    if set(report["models"]) != set(models):
        raise ValueError("The overview requires all three registered model reports")
    panels = {
        model: report["models"][model]["panels"]["primary|" + REPORT_METRICS[model][0]]
        for model in models
    }
    regions = {tuple(panel["regions"]) for panel in panels.values()}
    if len(regions) != 1 or len(next(iter(regions))) != 1:
        raise ValueError("The overview requires one common primary region")
    region = next(iter(regions))[0]
    region_label = labels[region]
    synthetic = report.get("input_seals", {}).get("synthetic_fixture_only") is True
    out = Path(output)
    out.mkdir(exist_ok=False)

    with matplotlib.rc_context({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.linewidth": 0.7, "axes.labelcolor": "#273547",
        "xtick.color": "#273547", "ytick.color": "#273547",
    }):
        fig = plt.figure(figsize=(14.2, 6.1))
        layout = fig.add_gridspec(2, 8, width_ratios=(1, 1, 0.22, 1, 1, 0.22, 1, 1),
                                 left=0.055, right=0.978, bottom=0.20, top=0.69,
                                 wspace=0.23, hspace=0.43)
        axes = []
        contracts = {}
        for model, start in zip(models, (0, 3, 6)):
            panel = panels[model]
            stats = panel["absolute_error"]
            group = []
            full_retention = [
                stats["means"][panel["points"].index(f"{region}|L{layer}|K10")]
                for layer in LAYERS[model]
            ]
            if len(set(full_retention)) != 1:
                raise ValueError(f"{model} has different full-retention means across the selected layers")
            baseline = full_retention[0]
            for i, layer in enumerate(LAYERS[model]):
                row, col = divmod(i, 2)
                ax = fig.add_subplot(layout[row, start + col],
                                     sharey=group[0] if group else None)
                group.append(ax)
                axes.append(ax)
                indices = [panel["points"].index(f"{region}|L{layer}|K{k}") for k in range(11)]
                if baseline is not None:
                    ax.axhline(baseline, color="#8793a0", linewidth=0.9, linestyle="--", zorder=1)
                _curve(ax, stats, indices, f"Layer {layer}", COLORS[i])
                ax.spines[["top", "right"]].set_visible(False)
                ax.spines[["left", "bottom"]].set_color("#8793a0")
                ax.grid(axis="y", color="#e2e7ec", linewidth=0.55)
                ax.set_axisbelow(True)
                ax.set_xlim(-0.02, 1.02)
                ax.set_xticks([0, 0.5, 1])
                ax.tick_params(labelbottom=row == 1, labelleft=col == 0, length=3, pad=3)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.set_title(f"Layer {layer}", loc="left", fontsize=9, fontweight="medium", pad=5)
                if panel["paired_sources"] == 0:
                    ax.text(0.5, 0.5, "No paired estimates", transform=ax.transAxes,
                            ha="center", va="center", color="#526375")
            center = (group[0].get_position().x0 + group[1].get_position().x1) / 2
            fig.text(center, 0.835, MODEL_LABELS[model], ha="center",
                     fontsize=13, fontweight="bold", color="#182434")
            fig.text(center, 0.79, units[model], ha="center", fontsize=9, color="#273547")
            fig.text(center, 0.754,
                     f"{panel['paired_sources']:,} paired  /  {panel['qualified_sources']:,} eligible",
                     ha="center", fontsize=8, color="#526375")
            contracts[model] = {
                "metric": panel["metric"], "region": region, "layers": list(LAYERS[model]),
                "paired_sources": panel["paired_sources"], "eligible_sources": panel["qualified_sources"],
                "requested_sources": panel["requested_sources"],
                "full_retention_mean": baseline, "status": panel["status"],
            }

        fig.text(0.055, 0.948, "BridgeData V2 visual-token masking", color="#182434",
                 fontsize=19, weight="bold", va="top")
        fig.text(0.056, 0.878, f"{region_label}  ·  {report['requested_source_identities']:,} indexed episodes",
                 fontsize=9.5, color="#526375")
        fig.text(0.515, 0.124, "Nominal fraction of ROI tokens retained", ha="center",
                 fontsize=10, color="#273547")
        fig.text(0.055, 0.077, "Mean and pointwise 95% paired-source CI. Dashed line shows full retention.",
                 fontsize=8, color="#526375")
        fig.text(0.055, 0.043, "Each model has its own vertical scale. Qwen layer 0 masking precedes DeepStack reinjection.",
                 fontsize=8, color="#526375")
        if synthetic:
            fig.text(0.5, 0.5, "SYNTHETIC FIXTURE", ha="center", va="center",
                     rotation=25, fontsize=32, color="#777777", alpha=0.12)
        layout_audit = _alignment(fig, axes)
        boxes = layout_audit["axes_rectangles_points"]
        tolerance = layout_audit["tolerance_pt"]
        rows = ((0, 1, 4, 5, 8, 9), (2, 3, 6, 7, 10, 11))
        columns = tuple((index, index + 2) for index in rows[0])
        for positions, coordinate in ((rows, 1), (columns, 0)):
            for parts in positions:
                offsets = [boxes[i][coordinate] for i in parts]
                if max(offsets) - min(offsets) > tolerance:
                    raise ValueError("The overview grid has misaligned rows or columns")
        layout_audit.update(rows=rows, columns=columns)
        fig.savefig(out / "bridge_primary_overview.png", dpi=300, facecolor="white")
        fig.savefig(out / "bridge_primary_overview.pdf", facecolor="white")
        fig.savefig(out / "bridge_primary_overview.svg", facecolor="white")
        plt.close(fig)

    contract = {
        "report_sha256": report_sha256, "models": contracts, "statistic": "absolute_error",
        "smoothing": False, "missing_imputed": False, "synthetic_fixture_only": synthetic,
        "alignment": layout_audit,
    }
    with (out / "OVERVIEW_CONTRACT.json").open("x", encoding="utf-8") as stream:
        json.dump(contract, stream, indent=2, allow_nan=False)
    return 12


def report_artifact(manifest, name):
    entries = [record for record in manifest["files"] if record["path"] == name]
    if manifest.get("schema") != "formal1500_report_artifact_manifest_v1" or len(entries) != 1:
        raise ValueError(f"Missing or ambiguous report artifact descriptor: {name}")
    return entries[0]


def load_verified_report(report_path):
    manifest = json.loads((report_path.parent / "ARTIFACT_MANIFEST.json").read_bytes())
    entry = report_artifact(manifest, report_path.name)
    body = report_path.read_bytes()
    if entry["bytes"] != len(body) or entry["sha256"] != sha256(body).hexdigest():
        raise ValueError("Report file differs from the saved diagnostic artifact manifest")
    report = json.loads(body)
    if report.get("schema") != "formal1500_report_v1":
        raise ValueError("Unknown diagnostic report format")
    return report, manifest, entry["sha256"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Replot a saved BridgeV2 diagnostic report.")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New directory for PNG and PDF figures")
    parser.add_argument("--overview", action="store_true", help="Draw the paper-style four-layer overview")
    args = parser.parse_args(argv)
    report, _, report_sha = load_verified_report(args.report.resolve(strict=True))
    if args.overview:
        panels = plot_primary_overview(report, args.output, report_sha)
        print(json.dumps({"figures": 1, "panels": panels, "output": str(args.output)}))
        return
    contracts = plot_report(report,args.output)
    print(json.dumps({"figures":len(contracts),"output":str(args.output)}))


if __name__ == "__main__":
    main()

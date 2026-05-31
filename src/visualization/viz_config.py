# viz_config.py
VIZ_PARAMS = {
    "gt_bbox_colors": {
        "person": "tab:green",
        "standing": "tab:orange",
        "lying": "tab:red",
        "seated": "tab:green",
        "seated_ground" : "#00ffb9",
        "clothing": "#110deee2", #blue (darker cyan)
        "bag": "#9B149B", #righ purple
    },
    "pred_bbox_colors": {
        "person": "#00ff00", #bright green
        "clothing": "deepskyblue",
        "bag": "mediumorchid",
        "seated": "#00ff00", #bright green
        "seated_ground": "tab:cyan",
        "standing":"#C76E00",
        "lying":"#8B0000",
    },

    "bbox_comparison": {
        "human-aligned": "tab:green",
        "axis-aligned": "tab:orange",
        "radius-aligned": "tab:blue",
    },
    
    "bbox_lines": {
        "thickness": 15,
    },
    "legend": {
        "linewidth": 15,
    },
    
    "label_prefix": "GT: "
}

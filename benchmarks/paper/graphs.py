"""The 16 graphs of the attention experiments (Table 1, Appendix J, the hardware counters),
by name, with the dataset config each one is loaded from. Ordered by edge count."""

ATTENTION_GRAPHS = [
    ("citeseer", "configs/datasets/secondary/citeseer.yaml"),
    ("cora", "configs/datasets/secondary/cora.yaml"),
    ("pubmed", "configs/datasets/secondary/pubmed.yaml"),
    ("city-roads-M", "configs/datasets/main/city_roads_m.yaml"),
    ("city-roads-L", "configs/datasets/graphland_remaining/city_roads_l.yaml"),
    ("artnet-exp", "configs/datasets/graphland_remaining/artnet_exp.yaml"),
    ("tolokers-2", "configs/datasets/main/tolokers_2.yaml"),
    ("ogbn-arxiv", "configs/datasets/main/ogbn_arxiv.yaml"),
    ("city-reviews", "configs/datasets/main/city_reviews.yaml"),
    ("twitch-views", "configs/datasets/main/twitch_views.yaml"),
    ("web-fraud", "configs/datasets/graphland_remaining/web_fraud.yaml"),
    ("hm-categories", "configs/datasets/main/hm_categories.yaml"),
    ("avazu-ctr", "configs/datasets/main/avazu_ctr.yaml"),
    ("pokec-regions", "configs/datasets/graphland_remaining/pokec_regions.yaml"),
    ("ogbn-proteins", "configs/datasets/secondary/ogbn_proteins.yaml"),
    ("ogbn-products", "configs/datasets/main/ogbn_products.yaml"),
]

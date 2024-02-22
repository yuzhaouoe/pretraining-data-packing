SUBSET_NAMES = [
    "RedPajamaCommonCrawl",
    "RedPajamaC4",
    "RedPajamaGithub",
    "RedPajamaBook",
    "RedPajamaArXiv",
    "RedPajamaWikipedia",
    "RedPajamaStackExchange",
]
SUBSET_WEIGHTS = {
    "RedPajamaCommonCrawl": 0.522,
    "RedPajamaC4": 0.267,
    "RedPajamaGithub": 0.052,
    "RedPajamaBook": 0.042,
    "RedPajamaArXiv": 0.046,
    "RedPajamaWikipedia": 0.038,
    "RedPajamaStackExchange": 0.033
}
SUBSET_SPLIT_NUMS = {
    "RedPajamaCommonCrawl": 86,
    "RedPajamaC4": 153,
    "RedPajamaGithub": 10,
    "RedPajamaBook": 1,
    "RedPajamaArXiv": 1,
    "RedPajamaWikipedia": 14,
    "RedPajamaStackExchange": 15,
}

WANDB_KEY = None
WANDB_PROJECT = None
WANDB_ENTITY = None
HUGGINGFACE_TOKEN = None

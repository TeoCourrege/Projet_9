"""Vérifie l'environnement : imports des bibliothèques clés et configuration (.env).

Usage : python scripts/check_env.py
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Équivalents actuels des imports de l'énoncé (LangChain 1.x et mistralai 3.x ont changé de chemins) :
#   from langchain.vectorstores import FAISS          -> langchain_community.vectorstores
#   from langchain.embeddings import HuggingFaceEmbeddings -> langchain_huggingface
#   from mistral import MistralClient                 -> from mistralai.client import Mistral
CHECKS = [
    ("faiss", None),
    ("langchain_community.vectorstores", "FAISS"),
    ("langchain_huggingface", "HuggingFaceEmbeddings"),
    ("langchain_mistralai", "ChatMistralAI"),
    ("mistralai.client", "Mistral"),
    ("sentence_transformers", "SentenceTransformer"),
    ("fastapi", "FastAPI"),
    ("pandas", None),
    ("bs4", "BeautifulSoup"),
]
OPTIONAL = [("ragas", "evaluate"), ("pytest", None)]


def check(module_name: str, attr: str | None) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
        if attr:
            getattr(module, attr)
        version = getattr(importlib.import_module(module_name.split(".")[0]), "__version__", "")
        return True, version
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    print(f"Python {sys.version.split()[0]}")
    ok = True
    for module_name, attr in CHECKS + OPTIONAL:
        success, info = check(module_name, attr)
        required = (module_name, attr) in CHECKS
        ok &= success or not required
        label = f"{module_name}{'.' + attr if attr else ''}"
        mark = "OK " if success else ("ERR" if required else "-- ")
        print(f"  [{mark}] {label:<45} {info}")

    from puls_rag.config import Settings

    settings = Settings()
    print("\nConfiguration :")
    print(f"  ville                 {settings.city}")
    print(f"  MISTRAL_API_KEY       {'définie' if settings.mistral_api_key else 'ABSENTE (requise pour /ask)'}")
    print(f"  ADMIN_TOKEN           {'défini' if settings.admin_token else 'absent (/rebuild désactivé)'}")
    print(f"  index FAISS           {'présent' if (settings.index_dir / 'index.faiss').exists() else 'absent : lancer scripts/build_index.py'}")
    print("\nEnvironnement OK" if ok else "\nDes imports obligatoires échouent : pip install -r requirements.txt")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

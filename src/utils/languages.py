DEFAULT_LANGUAGE = "en"

LANGUAGES: dict[str, dict] = {
    "en": {
        "native": "English",
        "name": "English",
        "labels": {
            "introduction": "Introduction",
            "conclusion": "Conclusion",
            "insights": "Insights",
            "summary": "Summary",
            "sources": "Sources",
            "closing": "Thank you so much for your help!",
            "no_sources": "I could not find any sources for this question, so I cannot answer it reliably.",
            "no_sources_section": "No sources could be gathered for this focus area, so there is nothing reliable to report.",
            "figure_elsewhere": (
                "The figure {figure} is not in the source(s) cited here ({cited}), but source(s) {found} contain it. "
                "Cite that source next to the figure, or remove the figure."),
            "figure_missing": (
                "The figure {figure} is not in the source(s) cited here ({cited}) or in any other source. "
                "Remove it, or replace it with what the sources say."),
        },
    },
    "tr": {
        "native": "Türkçe",
        "name": "Turkish (Türkçe)",
        "labels": {
            "introduction": "Giriş",
            "conclusion": "Sonuç",
            "insights": "Bulgular",
            "summary": "Özet",
            "sources": "Kaynaklar",
            "closing": "Yardımınız için çok teşekkür ederim!",
            "no_sources": "Bu soru için kaynak bulamadım, bu yüzden güvenilir bir yanıt veremiyorum.",
            "no_sources_section": "Bu odak alanı için kaynak toplanamadı, bu yüzden raporlanacak güvenilir bir bilgi yok.",
            "figure_elsewhere": (
                "{figure} rakamı burada atıf yapılan kaynaklarda ({cited}) yok, ama {found} kaynaklarında var. "
                "Rakamın yanında o kaynağa atıf yap ya da rakamı çıkar."),
            "figure_missing": (
                "{figure} rakamı burada atıf yapılan kaynaklarda ({cited}) ve başka hiçbir kaynakta yok. "
                "Rakamı çıkar ya da kaynakların söylediğiyle değiştir."),
        },
    },
}


def language_options() -> list[dict]:
    return [{"code": code, "name": info["native"]} for code, info in LANGUAGES.items()]


def normalize_language(code: str | None) -> str:
    code = (code or "").strip().lower()
    return code if code in LANGUAGES else DEFAULT_LANGUAGE


def language_name(code: str | None) -> str:
    return LANGUAGES[normalize_language(code)]["name"]


def label(code: str | None, key: str) -> str:
    return LANGUAGES[normalize_language(code)]["labels"][key]


def language_instruction(code: str | None) -> str:
    return (
        f"Write your entire response in {language_name(code)}. "
        "Keep proper names, URLs and citation markers such as [1] unchanged."
    )

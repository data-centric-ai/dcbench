def baseline(id: str, summary: str):
    def _baseline(fn: callable):
        fn.id = id
        fn.attributes = {"summary": summary}
        return fn

    return _baseline

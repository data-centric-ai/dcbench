def solver(id: str, summary: str):
    def _solver(fn: callable):
        fn.id = id
        fn.attributes = {"summary": summary}
        return fn

    return _solver

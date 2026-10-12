"""Semantic analysis helpers called by the parser while it builds nodes.

    typing.add_type          type an expression node
    conversions.*            promotions, usual arithmetic conversions, casts
    arith.new_add/new_sub    pointer arithmetic; compound assignment lowering
    constexpr.eval_const     integer constant expressions
    initializers.*           initializer lowering / static data serialization
"""

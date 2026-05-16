#mapping from natural language verbs to tool names
#when agent says "ill search for X" we need to map
#"search" to the web_search tool. this dict is the bridge
VERB_TO_TOOL = {
    # web_search synonyms
    "search": "web_search",
    "look up": "web_search",
    "google": "web_search",
    "find online": "web_search",
    "query": "web_search",
    "look for": "web_search",
    "retrieve": "web_search",
    # notes_write synonyms
    "write": "notes_write",
    "save": "notes_write",
    "note": "notes_write",
    "record": "notes_write",
    "jot down": "notes_write",
    "store": "notes_write",
}
# All known tools
KNOWN_TOOLS = {"web_search", "notes_write"}
# Omega

Omega is a personal agent that lives on your machine and keeps working with you across sessions. It maintains persistent conversations so you can pick up where you left off, a personal memory layer for durable facts about you and your work, and a knowledge base for documents, URLs, and code you want it to be able to search and reference later.

You choose the LLM provider. Omega keeps its database and local memory on your machine.

## Install

Requirements:

- Python 3.11+
- Docker with Docker Compose running
- An API key for Openrouter, Anthropic, or an OpenAI-compatible provider

```bash
pip install omega-agent
omega setup
```

`omega setup` starts Omega's bundled database, prepares its schema, asks for your LLM provider settings, verifies the connection, and checks the local embedding model.

## Start chatting
```bash
omega --help
```

That lists all the omega commands.

```bash
omega
```

That opens Omega's full-screen terminal interface.

Use the simpler scrolling chat instead:

```bash
omega --cli
```

Resume your most recent conversation:

```bash
omega --continue
```

Start a clean session:

```bash
omega new
```

Inside Omega TUI, type `/commands` to see the available chat commands. `/quit` exits.

## What Omega keeps

Omega has three seperate kinds of context:

- Conversation history: your saved chat sessions.
- Personal memory: durable facts you directly share, such as preferences or ongoing work
- Knowledge base: URLs, notes, and code you deliberately add for later reference.

## Add knowledge

```bash
omega ingest url "https://example.com/article"
omega ingest text "Notes I want Omega to search later"
omega ingest code --file ./main.py
```

Check queued or processed material:

```bash
omega kb list
omega kb status <item-id>
```

## Check omega's installation

```bash
omega doctor
```

This checks the bundled database, pgvector, embedding model, provider configuration, and local memory directory.

## Privacy

Omega stores its database and memory locally. Requests sent to an LLM go to the provider you configure. Those may include your chats, memory, knowledge base items. Your API key stays in your local configuration and is not in printed in Omega's normal output.
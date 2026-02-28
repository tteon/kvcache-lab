"""Shared infrastructure for trace collection across graph memory systems."""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Config constants ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GPU_ENDPOINT = os.getenv("GPU_ENDPOINT", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
GPU_MODEL = os.getenv("GPU_MODEL", "gpt-4o-mini")
GPU_API_KEY = os.getenv("GPU_API_KEY", OPENAI_API_KEY)

# Preferred runtime knobs for collectors:
# - LLM_* overrides take precedence.
# - GPU_* remains supported for backward compatibility with existing .env files.
LLM_API_BASE = os.getenv("LLM_API_BASE", GPU_ENDPOINT)
LLM_MODEL = os.getenv("LLM_MODEL", GPU_MODEL)
LLM_API_KEY = os.getenv("LLM_API_KEY", GPU_API_KEY)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRACES_DIR = PROJECT_ROOT / "data" / "traces"

# --- Test corpus: 50 factual statements with entities and relationships ---
# Items 0-9: original corpus (preserved for backward compatibility with prior traces)
# Items 10-49: expanded corpus across 5 domains with entity overlap clusters
#   - Elon Musk cluster (items 3, 8, 10-12): prefix-breaking entity reuse
#   - Nobel Prize cluster (items 0, 13-16): shared concept across fields
#   - Computing pioneers cluster (items 5, 7, 17-20): overlapping entity graph
#   - Domains: science, technology, geography, history, business/culture
TEST_CORPUS = [
    # --- Original 10 items (0-9) ---
    "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize.",
    "Albert Einstein developed the theory of general relativity, one of the two pillars of modern physics. He was born in Ulm, Germany in 1879.",
    "The Python programming language was created by Guido van Rossum and first released in 1991. It emphasizes code readability.",
    "Tesla Inc., founded by Martin Eberhard and Marc Tarpenning, is headquartered in Austin, Texas. Elon Musk joined as chairman in 2004.",
    "The Great Wall of China stretches over 13,000 miles and was built over many centuries starting from the 7th century BC.",
    "Ada Lovelace is often regarded as the first computer programmer. She worked with Charles Babbage on the Analytical Engine.",
    "The Amazon River is the largest river by volume of water flow in the world. It flows through Brazil, Peru, and Colombia.",
    "Alan Turing proposed the concept of the Turing machine in 1936, which became the foundation of modern computer science.",
    "SpaceX, founded by Elon Musk in 2002, developed the Falcon 9 rocket and the Dragon spacecraft for NASA missions.",
    "The Mediterranean Sea is connected to the Atlantic Ocean through the Strait of Gibraltar and borders Europe, Africa, and Asia.",
    # --- Elon Musk cluster (10-12) ---
    "Elon Musk co-founded Neuralink in 2016 to develop brain-computer interface technology. The company implanted its first device in a human patient in 2024.",
    "The Boring Company, founded by Elon Musk in 2016, builds underground transportation tunnels. Its first commercial project was the Las Vegas Convention Center Loop.",
    "Elon Musk acquired Twitter in October 2022 for approximately $44 billion and rebranded the platform to X in July 2023.",
    # --- Nobel Prize cluster (13-16) ---
    "Niels Bohr received the Nobel Prize in Physics in 1922 for his contributions to understanding atomic structure and quantum theory.",
    "Dorothy Hodgkin won the Nobel Prize in Chemistry in 1964 for determining the structures of important biochemical substances using X-ray crystallography.",
    "Martin Luther King Jr. was awarded the Nobel Peace Prize in 1964 for his nonviolent resistance to racial prejudice in the United States.",
    "Tu Youyou shared the Nobel Prize in Physiology or Medicine in 2015 for discovering artemisinin, a drug that significantly reduced malaria mortality.",
    # --- Computing pioneers cluster (17-20) ---
    "Grace Hopper developed the first compiler for a computer programming language and popularized the term 'debugging' in computing.",
    "John von Neumann designed the architecture that became the basis for most modern computers. He also contributed to game theory and quantum mechanics.",
    "Dennis Ritchie created the C programming language at Bell Labs in 1972 and co-developed the Unix operating system with Ken Thompson.",
    "Tim Berners-Lee invented the World Wide Web in 1989 while working at CERN. He also founded the World Wide Web Consortium to develop web standards.",
    # --- Science (21-27) ---
    "Isaac Newton published Principia Mathematica in 1687, establishing the laws of motion and universal gravitation that dominated physics for over two centuries.",
    "Rosalind Franklin's X-ray diffraction images of DNA were crucial to discovering the double helix structure. She worked at King's College London.",
    "The Hubble Space Telescope was launched in 1990 and has made over 1.5 million observations. It orbits Earth at about 547 kilometers altitude.",
    "CRISPR-Cas9 gene editing technology was developed by Jennifer Doudna and Emmanuelle Charpentier. They received the Nobel Prize in Chemistry in 2020.",
    "Charles Darwin published On the Origin of Species in 1859 after his voyage on HMS Beagle. His theory of natural selection transformed biology.",
    "The Large Hadron Collider at CERN is the world's largest particle accelerator, located beneath the France-Switzerland border near Geneva.",
    "Nikola Tesla invented the alternating current induction motor and contributed to the development of the modern AC electricity supply system.",
    # --- Technology (28-33) ---
    "Linux was created by Linus Torvalds in 1991 as a free open-source operating system kernel. It now powers most of the world's servers and supercomputers.",
    "Google was founded by Larry Page and Sergey Brin in 1998 while they were PhD students at Stanford University. Its search engine processed over 8.5 billion queries per day by 2024.",
    "The first iPhone was released by Apple in June 2007, designed under the leadership of Steve Jobs. It revolutionized the smartphone industry.",
    "NVIDIA was founded by Jensen Huang, Chris Malachowsky, and Curtis Priem in 1993. Its GPUs became essential for AI and deep learning workloads.",
    "Amazon Web Services launched in 2006, pioneering cloud computing infrastructure. It became the largest cloud provider by market share.",
    "OpenAI was founded in December 2015 as a nonprofit AI research lab. It released GPT-3 in 2020 and ChatGPT in November 2022.",
    # --- Geography (34-39) ---
    "The Nile River flows northward through eleven countries in Africa and is traditionally considered the longest river in the world at about 6,650 km.",
    "Mount Everest, located in the Himalayas on the border of Nepal and Tibet, stands at 8,849 meters as the highest point on Earth.",
    "The Sahara Desert covers approximately 9.2 million square kilometers across North Africa, making it the largest hot desert in the world.",
    "Japan consists of 6,852 islands in the Pacific Ocean. Its four largest islands are Honshu, Hokkaido, Kyushu, and Shikoku.",
    "The Panama Canal connects the Atlantic and Pacific Oceans across the Isthmus of Panama. It was completed in 1914 after a decade of construction.",
    "Lake Baikal in Siberia, Russia is the deepest lake in the world at 1,642 meters. It contains about 20% of the world's unfrozen surface fresh water.",
    # --- History (40-44) ---
    "The Roman Empire at its peak under Emperor Trajan around 117 AD controlled territory spanning from Britain to Mesopotamia.",
    "The printing press was invented by Johannes Gutenberg around 1440 in Mainz, Germany. It enabled the mass production of books and accelerated the spread of knowledge.",
    "The Apollo 11 mission landed the first humans on the Moon on July 20, 1969. Neil Armstrong and Buzz Aldrin walked on the lunar surface while Michael Collins orbited above.",
    "The Silk Road was an ancient network of trade routes connecting China to the Mediterranean. It facilitated the exchange of goods, culture, and ideas for over 1,500 years.",
    "The Industrial Revolution began in Britain in the late 18th century, transforming manufacturing from hand production to machine-based processes.",
    # --- Business and culture (45-49) ---
    "Microsoft was founded by Bill Gates and Paul Allen in 1975. Its Windows operating system became the dominant platform for personal computers.",
    "The Tokyo Stock Exchange is the largest stock exchange in Asia by market capitalization. It merged with the Osaka Securities Exchange in 2013 to form Japan Exchange Group.",
    "Samsung Electronics, headquartered in Suwon, South Korea, is the world's largest manufacturer of memory chips and smartphones by unit sales.",
    "The FIFA World Cup is the most widely viewed sporting event in the world. The 2022 tournament in Qatar attracted an estimated 5 billion viewers.",
    "Netflix was founded by Reed Hastings and Marc Randolph in 1997 as a DVD rental service. It launched its streaming platform in 2007 and had over 260 million subscribers by 2024.",
]


def messages_to_input_text(messages: list[dict]) -> str:
    """Convert an OpenAI-style messages array to a flat text string for trace format.

    Concatenates all message roles and contents into a single string,
    matching how prefix_analysis.py tokenizes the 'input' field.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Handle multimodal content blocks
            text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
            content = " ".join(text_parts)
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


class TraceLogger:
    """Writes JSONL traces compatible with lmcache-agent-trace/prefix_analysis.py.

    Output format per line:
        {"timestamp": <unix_us>, "input": "<text>", "output": "<text>", "session_id": "<id>"}
    """

    def __init__(self, output_path: str | Path, session_id: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self._file = open(self.output_path, "w", encoding="utf-8")

    def log(self, input_text: str, output_text: str, **metadata) -> None:
        """Write a single trace entry.

        Extra keyword arguments (model, prompt_tokens, completion_tokens, etc.)
        are merged into the JSONL line.  prefix_analysis.py only reads 'input'
        and 'output', so additional fields are safely ignored by the analyzer.
        """
        entry = {
            "timestamp": int(time.time() * 1_000_000),
            "input": input_text,
            "output": output_text,
            "session_id": self.session_id,
        }
        if metadata:
            entry.update(metadata)
        self._file.write(json.dumps(entry) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

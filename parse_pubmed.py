import os
import gzip
import glob
import xml.etree.ElementTree as ET

def parse_pubmed_file(file_path):
    with gzip.open(file_path, 'rb') as f:
        tree = ET.parse(f)
        root = tree.getroot()
        
        articles = []
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID")
            title = article.findtext(".//ArticleTitle")
            abstract = article.findtext(".//AbstractText")
            
            if title and abstract:
                full_text = f"PMID: {pmid}\nTITLE: {title}\nABSTRACT: {abstract}"
                articles.append(full_text)
        
        return articles

def load_pubmed_folder(folder_path="data/pubmed_data", max_files=5):
    files = sorted(glob.glob(os.path.join(folder_path, "*.xml.gz")))[:max_files]
    all_articles = []
    
    for f in files:
        print(f"Parsing {f} ...")
        articles = parse_pubmed_file(f)
        all_articles.extend(articles)
    
    return all_articles

if __name__ == "__main__":
    docs = load_pubmed_folder()
    print(f"Loaded {len(docs)} documents.")
    
    # Save as raw txt for later
    os.makedirs("data/parsed", exist_ok=True)
    with open("data/parsed/pubmed_texts.txt", "w", encoding="utf-8") as out:
        out.write("\n\n---\n\n".join(docs))

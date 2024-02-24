import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    prob_page = dict() 
    num_files = len(corpus) 
    num_links = len(corpus[page])

    if num_links!=0: 
        rand_prob = (1-damping_factor)/num_files
        spec_prob = damping_factor/num_files
    else: 
        rand_prob = (1-damping_factor)/num_files
        spec_prob = 0
    
    for files in corpus:
        if len(corpus[page]) == 0: 
            prob_page[files] = 1/num_files
        else: 
            if files not in corpus[page]: 
                prob_page[files] = rand_prob 
            else: 
                prob_page[files] = spec_prob + rand_prob 

    return prob_page


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_rank = dict() 
    for page in corpus: 
        page_rank[page] = 0 
    
    sample = None

    for _ in range(n): 
        if sample == None: 
            choices = list(corpus.keys())
            sample = random.choice(choices) 
            page_rank[sample] += 1 
        else:
            next_SP = transition_model(corpus, sample, damping_factor)
            choices = list(next_SP.keys())
            weights = [next_SP[keys] for keys in choices]
            sample = random.choices(choices, weights).pop()
            page_rank[sample] += 1 
    
    
    for key, value in page_rank.items(): 
        page_rank = {key: value}

    if sum(round(page_rank.values()), 5) != 5: 
        print("Error sum doesn't equal to 5") 
    else: 
        print("Sum of sample page rank values: ", round(sum(page_rank.values()))) 

    return page_rank






def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    iterate_PR = dict() 
    N = len(corpus) 
    
    for pages in corpus: 
        iterate_PR[pages] = 1/N 
    
    changes = 1
    iterations = 1
    while changes >= 0.001: 
        changes = 0 
        previous_state = iterate_PR.copy()  
        for page in iterate_PR: 
            parents = [link for link in corpus if page in corpus[link]] 
            first = ((1 - damping_factor)/N) 
            second = []
            if len(parents) != 0: 
                for parent in parents: 
                    num_links = len(corpus[parent]) 
                    val = previous_state[parent]/num_links 
                    second.append(val) 

            second = sum(second) 
            iterate_PR[page] = first + (damping_factor * second)
            new_change = abs(iterate_PR[page] - previous_state[page])
            if changes < new_change: 
                change = new_change 
            
        iterations += 1

    dictsum = sum(iterate_PR.values()) 
    iterate_PR = {key: value/dictsum for key, value in iterate_PR.items()} 
    print(f"\nPageRank value stable after {iterations}") 
    print("Sum of iterate_pagerank values: ", round(dictsum, 10))
    return iterate_PR


if __name__ == "__main__":
    main()

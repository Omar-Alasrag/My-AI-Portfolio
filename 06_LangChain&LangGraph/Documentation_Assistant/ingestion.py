import asyncio

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_tavily import TavilyCrawl
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


class DocumentationIngestor:
    def __init__(
        self, persist_directory="chroma_db", index_name="documentation-assistant"
    ):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            output_dimensionality=1024,
            max_retries=6,
            chunk_size=3,
        )

        self.vectorstore = Chroma(
            persist_directory=persist_directory, embedding_function=self.embeddings
        )

        self.tavily_crawl = TavilyCrawl()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )

    async def _ingest_chunks_async(self, chunk_batches, max_concurrency=5):
        semaphore = asyncio.Semaphore(max_concurrency)

        async def safe_area(batch, batch_num):
            async with semaphore:
                try:
                    return await self.vectorstore.aadd_documents(batch)
                except Exception as e:
                    print(f"Exception for batch {batch_num}: - {e}")
                    return e

        tasks = [safe_area(batch, i) for i, batch in enumerate(chunk_batches)]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def ingest(
        self, url, max_depth=5, instructions="content about agent ai", batch_size=3
    ):
        print("################### INGESTION ###################")

        print(f"crawling {url}")
        res = self.tavily_crawl.invoke(
            {
                "url": url,
                "max_depth": max_depth,
                "instructions": instructions,
            }
        )

        all_docs = [
            Document(page_content=r["raw_content"], metadata={"source": r["url"]})
            for r in res["results"]
        ]

        print("splitting ...")
        chunks = self.splitter.split_documents(all_docs)
        print(f"length of chunks is {len(chunks)}")

        chunk_batches = [
            chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)
        ]

        print("ingesting ...")
        ingest_results = await self._ingest_chunks_async(
            chunk_batches, max_concurrency=5
        )

        n_success = len(
            [
                ele
                for ele in ingest_results
                if not isinstance(ele, Exception) and ele is not None
            ]
        )

        if n_success == len(chunk_batches):
            print("All Batches Indexed Successfully")
        else:
            print(f"The successful batches are {n_success}/{len(chunk_batches)}")

        print("ingestion completed")


if __name__ == "__main__":
    ingestor = DocumentationIngestor()

    asyncio.run(
        ingestor.ingest(
            url="https://python.langchain.com/docs/introduction/",
            instructions="content about agent ai",
        )
    )

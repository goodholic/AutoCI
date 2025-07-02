

import asyncio
import json
from typing import List, Dict, Any
from googlesearch import search
import requests
from bs4 import BeautifulSoup

class IntelligentInformationGatherer:
    """
    웹 및 외부 소스에서 정보를 수집하여 학습 데이터를 생성하는 시스템
    """

    def __init__(self):
        pass

    async def search_web_for_code(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        웹에서 코드 스니펫 및 관련 정보를 검색합니다.

        Args:
            query: 검색어
            num_results: 검색 결과 수

        Returns:
            검색된 코드 정보 리스트
        """
        loop = asyncio.get_event_loop()
        try:
            search_results = await loop.run_in_executor(
                None,
                lambda: list(search(query, num_results=num_results, stop=num_results, pause=2.0))
            )
        except Exception as e:
            print(f"웹 검색 중 오류 발생: {e}")
            return []
        
        code_snippets = []
        for url in search_results:
            if "stackoverflow.com" in url or "github.com" in url:
                try:
                    response = requests.get(url)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    code_blocks = soup.find_all('code')
                    for block in code_blocks:
                        code_snippets.append({
                            "source": url,
                            "query": query,
                            "code": block.get_text(),
                            "explanation": "Code snippet from the web."
                        })
                except Exception as e:
                    print(f"URL {url} 처리 중 오류 발생: {e}")
        return code_snippets

    async def scrape_godot_docs(self, url: str) -> List[Dict[str, Any]]:
        """
        Godot 공식 문서에서 정보를 스크래핑합니다.

        Args:
            url: 스크래핑할 Godot 문서 URL

        Returns:
            스크래핑된 정보 리스트
        """
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            docs_data = []
            # This is a simplified example. A real implementation would need to be
            # more robust to handle the specific structure of the Godot documentation.
            title = soup.find('h1').get_text() if soup.find('h1') else "Untitled"
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                docs_data.append({
                    "source": url,
                    "title": title,
                    "content": p.get_text()
                })
            return docs_data
        except Exception as e:
            print(f"Godot 문서 스크래핑 중 오류 발생: {e}")
            return []

    async def gather_and_process_csharp_code(self) -> None:
        """
        C# 코드 관련 정보를 수집하고 처리합니다.
        """
        queries = [
            "C# Godot best practices",
            "C# design patterns examples",
            "C# performance optimization tips"
        ]
        
        all_code_data = []
        for query in queries:
            results = await self.search_web_for_code(query)
            all_code_data.extend(results)

        with open("collected_csharp_code.json", "w", encoding="utf-8") as f:
            json.dump(all_code_data, f, indent=2, ensure_ascii=False)

    async def gather_and_process_godot_docs(self) -> None:
        """
        Godot 문서를 수집하고 처리합니다.
        """
        urls = [
            "https://docs.godotengine.org/en/stable/getting_started/step_by_step/scenes_and_nodes.html",
            "https://docs.godotengine.org/en/stable/getting_started/step_by_step/scripting_continued.html",
        ]
        
        all_docs_data = []
        for url in urls:
            results = await self.scrape_godot_docs(url)
            all_docs_data.extend(results)

        with open("collected_godot_docs.json", "w", encoding="utf-8") as f:
            json.dump(all_docs_data, f, indent=2, ensure_ascii=False)

# Singleton instance
_information_gatherer = None

def get_information_gatherer() -> IntelligentInformationGatherer:
    """
    IntelligentInformationGatherer의 싱글톤 인스턴스를 반환합니다.
    """
    global _information_gatherer
    if _information_gatherer is None:
        _information_gatherer = IntelligentInformationGatherer()
    return _information_gatherer

if __name__ == "__main__":
    async def main():
        gatherer = get_information_gatherer()
        await gatherer.gather_and_process_csharp_code()
        await gatherer.gather_and_process_godot_docs()
        print("C# code and Godot docs gathering and processing complete.")

    asyncio.run(main())
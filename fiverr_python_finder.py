#!/usr/bin/env python3
"""
Fiverr Python Jobs Finder
Программа для поиска заказов на Fiverr, связанных с разработкой на Python.
Использует Selenium для автоматизации браузера и опционально локальную нейросеть.
"""

import json
import time
import os
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Опциональный импорт для нейросети
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False


@dataclass
class Job:
    """Класс для представления заказа"""
    title: str
    description: str
    budget: str
    client_info: str
    url: str
    found_at: str
    relevance_score: Optional[float] = None


class NeuralFilter:
    """Класс для фильтрации заказов с помощью локальной нейросети"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        """
        Инициализация нейросети
        
        Args:
            model_name: Название модели для загрузки (по умолчанию Qwen2.5-3B)
        """
        if not NEURAL_AVAILABLE:
            print("⚠️  Transformers library not installed. Neural filtering disabled.")
            self.enabled = False
            return
            
        self.enabled = True
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
        try:
            print(f"🔄 Loading neural model: {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            if not torch.cuda.is_available():
                self.model = self.model.to('cpu')
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"⚠️  Failed to load model: {e}")
            self.enabled = False
    
    def evaluate_relevance(self, job_text: str) -> float:
        """
        Оценка релевантности заказа разработке на Python
        
        Args:
            job_text: Текст заказа
            
        Returns:
            float: Оценка релевантности от 0.0 до 1.0
        """
        if not self.enabled:
            return 0.5  # Возвращаем среднее значение если нейросеть не доступна
        
        prompt = f"""
Ты помощник для оценки релевантности заказов на разработку.
Оцени заказ по шкале от 0 до 1, где 1 - очень релевантный заказ на разработку на Python.

Заказ:
{job_text}

Оценка (только число от 0.0 до 1.0):
"""
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Извлекаем число из ответа
            import re
            numbers = re.findall(r'\d+\.?\d*', response)
            if numbers:
                score = float(numbers[-1])
                return min(max(score, 0.0), 1.0)
            return 0.5
        except Exception as e:
            print(f"⚠️  Neural evaluation error: {e}")
            return 0.5
    
    def is_relevant(self, job_text: str, threshold: float = 0.6) -> bool:
        """
        Проверка релевантности заказа
        
        Args:
            job_text: Текст заказа
            threshold: Порог релевантности
            
        Returns:
            bool: True если заказ релевантен
        """
        score = self.evaluate_relevance(job_text)
        return score >= threshold


class FiverrJobFinder:
    """Основной класс для поиска заказов на Fiverr"""
    
    def __init__(self, use_neural: bool = False, headless: bool = False):
        """
        Инициализация поисковика
        
        Args:
            use_neural: Использовать ли нейросеть для фильтрации
            headless: Запускать ли браузер в фоновом режиме
        """
        self.use_neural = use_neural
        self.neural_filter = NeuralFilter() if use_neural else None
        self.jobs: List[Job] = []
        self.headless = headless
        self.driver = None
        
    def setup_driver(self):
        """Настройка WebDriver для Chrome"""
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument("--headless=new")
        
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            print("✅ Browser initialized successfully!")
        except Exception as e:
            print(f"❌ Failed to initialize browser: {e}")
            print("💡 Make sure you have Chrome and ChromeDriver installed")
            raise
    
    def search_jobs(self, keywords: List[str] = None, max_jobs: int = 20):
        """
        Поиск заказов
        
        Args:
            keywords: Ключевые слова для поиска
            max_jobs: Максимальное количество заказов для сбора
        """
        if keywords is None:
            keywords = [
                "python development",
                "python developer",
                "python script",
                "python automation",
                "python bot",
                "python web scraping",
                "python django",
                "python flask",
                "python api"
            ]
        
        if not self.driver:
            self.setup_driver()
        
        base_url = "https://www.fiverr.com/search/gigs?query="
        
        for keyword in keywords:
            if len(self.jobs) >= max_jobs:
                break
                
            try:
                search_url = base_url + keyword.replace(" ", "%20")
                print(f"\n🔍 Searching for: {keyword}")
                
                self.driver.get(search_url)
                time.sleep(3)  # Ждем загрузки страницы
                
                # Пытаемся найти gigs (заказы/услуги)
                # Примечание: структура Fiverr может меняться, это примерный парсинг
                try:
                    gig_cards = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "[data-testid='gig-card']"))
                    )
                    
                    for card in gig_cards[:5]:  # Берем топ-5 с каждой страницы
                        if len(self.jobs) >= max_jobs:
                            break
                            
                        try:
                            title_elem = card.find_element(By.CSS_SELECTOR, "[data-testid='gig-title']")
                            title = title_elem.text.strip()
                            
                            # Пытаемся получить дополнительную информацию
                            try:
                                desc_elem = card.find_element(By.CSS_SELECTOR, ".description")
                                description = desc_elem.text.strip()
                            except:
                                description = ""
                            
                            try:
                                price_elem = card.find_element(By.CSS_SELECTOR, "[data-testid='gig-price']")
                                budget = price_elem.text.strip()
                            except:
                                budget = "Not specified"
                            
                            try:
                                client_elem = card.find_element(By.CSS_SELECTOR, "[data-testid='seller-name']")
                                client_info = client_elem.text.strip()
                            except:
                                client_info = "Unknown"
                            
                            # Ссылка на заказ
                            try:
                                link_elem = card.find_element(By.TAG_NAME, "a")
                                url = link_elem.get_attribute("href")
                            except:
                                url = search_url
                            
                            job_text = f"{title} {description}"
                            
                            # Оценка релевантности через нейросеть если включена
                            relevance_score = None
                            if self.neural_filter and self.neural_filter.enabled:
                                relevance_score = self.neural_filter.evaluate_relevance(job_text)
                                print(f"   📊 Relevance score: {relevance_score:.2f}")
                            else:
                                # Простая проверка по ключевым словам
                                python_keywords = ["python", "django", "flask", "fastapi", "scrapy", "selenium"]
                                relevance_score = sum(1 for kw in python_keywords if kw.lower() in job_text.lower()) / len(python_keywords)
                            
                            job = Job(
                                title=title,
                                description=description,
                                budget=budget,
                                client_info=client_info,
                                url=url,
                                found_at=datetime.now().isoformat(),
                                relevance_score=relevance_score
                            )
                            
                            self.jobs.append(job)
                            print(f"   ✅ Found: {title[:50]}...")
                            
                        except Exception as e:
                            print(f"   ⚠️  Error parsing card: {e}")
                            continue
                            
                except TimeoutException:
                    print(f"   ⚠️  No results found for '{keyword}' or page structure changed")
                    continue
                    
            except Exception as e:
                print(f"   ❌ Error searching for '{keyword}': {e}")
                continue
        
        print(f"\n📈 Total jobs found: {len(self.jobs)}")
    
    def save_to_file(self, filename: str = None):
        """
        Сохранение найденных заказов в файл
        
        Args:
            filename: Имя файла для сохранения
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fiverr_python_jobs_{timestamp}.json"
        
        if not filename.endswith(".json"):
            filename += ".json"
        
        data = {
            "search_date": datetime.now().isoformat(),
            "total_jobs": len(self.jobs),
            "jobs": [asdict(job) for job in self.jobs]
        }
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Saved {len(self.jobs)} jobs to {filename}")
        
        # Также сохраняем в текстовом формате для удобства чтения
        txt_filename = filename.replace(".json", ".txt")
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(f"Fiverr Python Jobs - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            for i, job in enumerate(self.jobs, 1):
                f.write(f"#{i}\n")
                f.write(f"Title: {job.title}\n")
                f.write(f"Description: {job.description}\n")
                f.write(f"Budget: {job.budget}\n")
                f.write(f"Client: {job.client_info}\n")
                f.write(f"Relevance Score: {job.relevance_score}\n")
                f.write(f"URL: {job.url}\n")
                f.write(f"Found at: {job.found_at}\n")
                f.write("-" * 80 + "\n\n")
        
        print(f"💾 Also saved text version to {txt_filename}")
        return filename
    
    def close(self):
        """Закрытие браузера"""
        if self.driver:
            self.driver.quit()
            print("👋 Browser closed")


def main():
    """Основная функция программы"""
    print("=" * 80)
    print("🐍 Fiverr Python Jobs Finder")
    print("=" * 80)
    print()
    
    # Настройки
    USE_NEURAL = False  # Установите в True если хотите использовать нейросеть
    HEADLESS = False    # Установите в True для запуска без интерфейса браузера
    MAX_JOBS = 20       # Максимальное количество заказов для сбора
    
    print(f"Configuration:")
    print(f"  - Neural Network: {'Enabled' if USE_NEURAL else 'Disabled'}")
    print(f"  - Headless Mode: {'On' if HEADLESS else 'Off'}")
    print(f"  - Max Jobs: {MAX_JOBS}")
    print()
    
    finder = None
    
    try:
        finder = FiverrJobFinder(use_neural=USE_NEURAL, headless=HEADLESS)
        finder.setup_driver()
        
        # Ключевые слова для поиска
        keywords = [
            "python development",
            "python developer needed",
            "python script",
            "python automation",
            "python bot telegram",
            "python web scraping",
            "python django developer",
            "python flask",
            "python api development",
            "python data scraping"
        ]
        
        finder.search_jobs(keywords=keywords, max_jobs=MAX_JOBS)
        
        if finder.jobs:
            finder.save_to_file()
        else:
            print("\n⚠️  No jobs found. Try adjusting search keywords or check your internet connection.")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Search interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if finder:
            finder.close()
    
    print("\n" + "=" * 80)
    print("✅ Search completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

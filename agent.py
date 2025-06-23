from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
import pandas as pd
import numpy as np
import anthropic

class AgentState(TypedDict):
    """Состояние агента"""
    messages: Annotated[list, add_messages]
    user_query: str
    profile: str
    search_performed: bool
    search_results: str

class Agent:
    """LangGraph агент с Anthropic API"""
    biases = ['стадность','излишняя самоуверенность']
    sex = {1: 'мужчина', 2: 'женщина'}
    tip = {1:'Москва или Санкт-Петербург',
       2: 'город - миллионник (численность людей)',
       3: 'городе ,где проживают 500-950 тыс. человек',
       4: 'городе ,где проживают 100-500 тыс. человек',
       5: 'городе ,где проживают до 100 тыс. человек',
       6: 'селе',
       7: 'неизвестно, что за город'}
    fo =  {1: 'Центральный федеральный округ',
       2: 'Северо-Западный федеральный округ',
       3: 'Южный федеральный округ',
       4: 'Северо-Кавказский федеральный округ',
       5: 'Приволжский федеральный округ',
       6: 'Уральский федеральный округ',
       7: 'Сибирский федеральный округ',
       8: 'Дальневосточный федеральный округ'}
    prof = {1: 'неработающий пенсионер',
       2: 'работающий пенсионер',
       3: 'неработающий учащийся, студент',
       4: 'работающий учащийся, студент',
       5: 'безработный',
       6: 'находящийся в декретном отпуске',
       7: 'работающий в найме',
       8: 'предприниматель',
       9: 'самозанятый',
       99: '-',
       999: '-'}
    dohod = {1:'очень хорошее',
       2: 'хорошее',
       3: 'среднее',
       4: 'плохое',
       5: 'очень плохое',
       99: 'неизвестно'}
    edu = {1:'неполное среднее образование',
       2: 'среднее образование (школа или ПТУ)',
       3: 'среднее специальное образование (техникум)',
       4: 'незаконченное высшее (с 3-го курса ВУЗа)',
       5: 'высшее образование',
       6: 'неизвестное образование',
       999: 'неизвестное образование'}
    
    def __init__(self, api_key: str, row: pd.Series, q_num:int, model: str = "claude-sonnet-4-20250514"):#claude-3-sonnet-20240229
        """
        Инициализация агента с Anthropic API
        
        Args:
            api_key: API ключ для Anthropic
            row: строка с демографическими данными
            model: модель Claude для использования
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.row = row        
        self.bias = np.random.choice(self.biases, 1)[0]
        self.q_num = q_num
        self.graph = self._create_graph()

    def extract_inflation_score(self, response: str) -> int:
        """Извлекает числовую оценку инфляции из ответа модели"""
        if self.q_num == 1:
            # Ищем число от 0 до 5 в ответе
            if "Серьезно вырастут" in response:
                return 5
            elif "Незначительно вырастут" in response:
                return 4
            elif "Останутся на нынешнем уровне" in response:
                return 3
            elif "Незначительно снизятся" in response:
                return 2
            elif "Серьезно снизятся" in response:
                return 1
            else:
                return 0
        else:
            # Ищем число от 0 до 3 в ответе
            if "Инфляция очень высокая" in response:
                return 3
            elif "Инфляция умеренная" in response:
                return 2
            elif "Инфляция незначительная" in response:
                return 1
            else:
                return 0

    @staticmethod
    def save_responses_to_csv(responses: list, filename: str):
        """Сохраняет список ответов с числовыми оценками в CSV файл"""
        df = pd.DataFrame(responses, columns=['response', 'inflation_score'])
        df.to_csv(filename, index=True, encoding='utf-8')
        print(f"Результаты сохранены в файл: {filename}")
        
    def _generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Генерация текста через Anthropic API"""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.7,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )
            return message.content[0].text
        except Exception as e:
            return f"Ошибка генерации: {str(e)}"

    def create_profile(self): 
        """Создает промпт на основе demographics_info и bias"""
        profile_prompt = f"""Представь, что сейчас 2023 года, ТЫ {self.prof[self.row['PROF']]} {self.sex[self.row['SEX']]} {self.row['AGE']} лет, проживающий в России ({self.fo[self.row['FO']]}) в {self.tip[self.row['TIP']]}. Твое материальное состояние можно охарактеризовать, как {self.dohod[self.row['DOHOD']]}. Ты получил {self.edu[self.row['EDU']]}. Тебе присуща {self.bias}.    
        """
        return profile_prompt

    def _create_graph(self):
        """Создает LangGraph граф"""
        
        # Определяем граф
        graph_builder = StateGraph(AgentState)
        
        # Добавляем nodes
        graph_builder.add_node("initialize_profile", self._initialize_profile_node)
        graph_builder.add_node("search", self._search_node)
        graph_builder.add_node("generate_response", self._generate_response_node)
        
        # Добавляем edges
        graph_builder.add_edge(START, "initialize_profile")
        graph_builder.add_edge("initialize_profile", "search")  # Обязательный поиск
        graph_builder.add_edge("search", "generate_response")
        graph_builder.add_edge("generate_response", END)
        
        return graph_builder.compile()

    def _initialize_profile_node(self, state: AgentState) -> AgentState:
        """Node для инициализации профиля"""
        profile = self.create_profile()
        return {
            **state,
            "profile": profile,
            "search_performed": False,
            "search_results": ""
        }
        
    def _search_node(self, state: AgentState) -> AgentState:
        """Node для поиска с обработкой ошибок"""
        search_query = self._create_search_query(state)
            
        return {
            **state,
            "search_performed": True,
            "search_results": search_query
            }

    def _create_search_query(self, state: AgentState) -> str:
        """Создает поисковый запрос"""
        query = f"Опиши текущую экономическую ситуацию для меня с учетом МОИХ демо-географических характеристик на момент 2023 года."
        system_prompt = f"""{state['profile']}\n. Отвечай от первого лица, учитывая свой профиль."""
        return self._generate_text(system_prompt, query)

    def _generate_response_node(self, state: AgentState) -> AgentState:
        """Node для генерации финального ответа"""
        system_prompt = f"""{state['profile']}\n{state['search_results']}\nОтвечай от первого лица, полностью вживаясь в роль описанного человека."""
        
        response = self._generate_text(system_prompt, state['user_query'])
        
        return {
            **state,
            "messages": [AIMessage(content=response)]
        }

    def process_query(self, query: str) -> str:
        """Обрабатывает запрос пользователя через LangGraph"""
        try:
            # Инициализируем состояние
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "user_query": query,
                "profile": "",
                "search_performed": False,
                "search_results": ""
            }
            
            # Запускаем граф
            result = self.graph.invoke(initial_state)
            
            # Возвращаем последнее сообщение
            if result.get("messages") and len(result["messages"]) > 0:
                return result["messages"][-1].content
            else:
                return "Не удалось получить ответ."
                
        except Exception as e:
            return f"Ошибка при обработке запроса: {str(e)}"


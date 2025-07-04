import os
import asyncio
import aiohttp
import json
import sqlite3
import re
import uuid
from typing import Optional, Dict, Any, List, AsyncGenerator, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path
import hashlib
import logging
import html

# FastAPI imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carregar variáveis de ambiente
load_dotenv()

# ===== MODELOS E CLASSES DO CHATBOT =====

class AIModel(Enum):
    """Modelos de IA disponíveis"""
    GEMINI = "Gemini 2.5 Pro"
    DEEPSEEK = "DeepSeek Reasoner"
    GROK = "Grok 3 Mini"

@dataclass
class Session:
    """Representa uma sessão de chat"""
    id: str
    name: str
    created_at: datetime
    last_accessed: datetime
    model: AIModel
    message_count: int = 0

class DatabaseManager:
    """Gerencia o banco de dados SQLite"""
    
    def __init__(self, db_path: str = "chatbot_sessions.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Inicializa o banco de dados com as tabelas necessárias"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Tabelas principais
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    last_accessed TIMESTAMP NOT NULL,
                    model TEXT NOT NULL,
                    message_count INTEGER DEFAULT 0,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    model TEXT NOT NULL,
                    tokens_used INTEGER DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS code_snippets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER NOT NULL,
                    language TEXT,
                    code TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    FOREIGN KEY (message_id) REFERENCES messages(id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    content_type TEXT,
                    file_path TEXT NOT NULL,
                    uploaded_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            ''')
            
            # Novas tabelas para as melhorias
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prompt_templates (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    prompt TEXT NOT NULL,
                    category TEXT,
                    is_public BOOLEAN DEFAULT 0,
                    created_by TEXT,
                    created_at TIMESTAMP NOT NULL,
                    usage_count INTEGER DEFAULT 0
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS message_favorites (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    user_note TEXT,
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (message_id) REFERENCES messages(id),
                    FOREIGN KEY (session_id) REFERENCES sessions(id),
                    UNIQUE(message_id, session_id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS session_tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    color TEXT DEFAULT '#3b82f6',
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id),
                    UNIQUE(session_id, tag)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            ''')
            
            # Índices para performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_code_message ON code_snippets(message_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(is_active)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_templates_category ON prompt_templates(category)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_favorites_session ON message_favorites(session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tags_session ON session_tags(session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tags_tag ON session_tags(tag)')
            
            conn.commit()
    
    def create_session(self, name: str, model: AIModel) -> Session:
        """Cria uma nova sessão"""
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO sessions (id, name, created_at, last_accessed, model, is_active)
                VALUES (?, ?, ?, ?, ?, 1)
            ''', (session_id, name, now, now, model.name))
            conn.commit()
        
        return Session(session_id, name, now, now, model)
    
    def get_sessions(self, active_only: bool = True) -> List[Session]:
        """Retorna todas as sessões"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            query = '''
                SELECT id, name, created_at, last_accessed, model, message_count
                FROM sessions
            '''
            if active_only:
                query += ' WHERE is_active = 1'
            query += ' ORDER BY last_accessed DESC'
            
            cursor.execute(query)
            
            sessions = []
            for row in cursor.fetchall():
                sessions.append(Session(
                    id=row[0],
                    name=row[1],
                    created_at=datetime.fromisoformat(row[2]),
                    last_accessed=datetime.fromisoformat(row[3]),
                    model=AIModel[row[4]],
                    message_count=row[5]
                ))
            
            return sessions
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Obtém uma sessão específica"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, name, created_at, last_accessed, model, message_count
                FROM sessions WHERE id = ? AND is_active = 1
            ''', (session_id,))
            
            row = cursor.fetchone()
            if row:
                return Session(
                    id=row[0],
                    name=row[1],
                    created_at=datetime.fromisoformat(row[2]),
                    last_accessed=datetime.fromisoformat(row[3]),
                    model=AIModel[row[4]],
                    message_count=row[5]
                )
            return None
    
    def update_session_access(self, session_id: str):
        """Atualiza o último acesso da sessão"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE sessions
                SET last_accessed = ?, message_count = message_count + 1
                WHERE id = ?
            ''', (datetime.now(), session_id))
            conn.commit()
    
    def update_session_model(self, session_id: str, model: AIModel):
        """Atualiza o modelo da sessão"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE sessions SET model = ? WHERE id = ?
            ''', (model.name, session_id))
            conn.commit()
    
    def soft_delete_session(self, session_id: str):
        """Soft delete de uma sessão"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE sessions SET is_active = 0 WHERE id = ?
            ''', (session_id,))
            conn.commit()
    
    def save_message(self, session_id: str, role: str, content: str, model: AIModel, tokens: int = 0) -> int:
        """Salva uma mensagem no banco"""
        escaped_content = html.escape(content)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO messages (session_id, role, content, timestamp, model, tokens_used)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (session_id, role, escaped_content, datetime.now(), model.name, tokens))
            conn.commit()
            return cursor.lastrowid
    
    def save_code_snippets(self, message_id: int, snippets: List[Tuple[str, str, int]]):
        """Salva snippets de código associados a uma mensagem"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for language, code, position in snippets:
                cursor.execute('''
                    INSERT INTO code_snippets (message_id, language, code, position)
                    VALUES (?, ?, ?, ?)
                ''', (message_id, language, code, position))
            conn.commit()
    
    def save_file(self, session_id: str, filename: str, content_type: str, file_path: str):
        """Salva informações de um arquivo"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO files (session_id, filename, content_type, file_path, uploaded_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, filename, content_type, file_path, datetime.now()))
            conn.commit()
    
    def get_session_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retorna todas as mensagens de uma sessão com status de favorito"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            query = '''
                SELECT m.id, m.role, m.content, m.timestamp, m.model,
                       CASE WHEN mf.id IS NOT NULL THEN 1 ELSE 0 END as is_favorite,
                       mf.user_note
                FROM messages m
                LEFT JOIN message_favorites mf ON m.id = mf.message_id AND mf.session_id = ?
                WHERE m.session_id = ?
                ORDER BY m.timestamp ASC
            '''
            if limit:
                query += f' LIMIT {limit}'
            
            cursor.execute(query, (session_id, session_id))
            
            messages = []
            for row in cursor.fetchall():
                content = html.unescape(row[2])
                
                message = {
                    "id": row[0],
                    "role": row[1],
                    "content": content,
                    "timestamp": row[3],
                    "model": row[4],
                    "is_favorite": bool(row[5]),
                    "favorite_note": row[6] if row[6] else "",
                    "code_snippets": []
                }
                
                # Buscar snippets de código
                cursor.execute('''
                    SELECT language, code, position
                    FROM code_snippets
                    WHERE message_id = ?
                    ORDER BY position
                ''', (row[0],))
                
                for snippet in cursor.fetchall():
                    message["code_snippets"].append({
                        "language": snippet[0],
                        "code": snippet[1],
                        "position": snippet[2]
                    })
                
                messages.append(message)
            
            return messages
    
    def search_messages(self, query: str, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Busca mensagens por conteúdo"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if session_id:
                cursor.execute('''
                    SELECT m.id, m.content, m.timestamp, s.name, m.model, m.session_id
                    FROM messages m
                    JOIN sessions s ON m.session_id = s.id
                    WHERE m.content LIKE ? AND m.session_id = ? AND s.is_active = 1
                    ORDER BY m.timestamp DESC
                    LIMIT 50
                ''', (f'%{query}%', session_id))
            else:
                cursor.execute('''
                    SELECT m.id, m.content, m.timestamp, s.name, m.model, m.session_id
                    FROM messages m
                    JOIN sessions s ON m.session_id = s.id
                    WHERE m.content LIKE ? AND s.is_active = 1
                    ORDER BY m.timestamp DESC
                    LIMIT 50
                ''', (f'%{query}%',))
            
            results = []
            for row in cursor.fetchall():
                content = html.unescape(row[1])
                results.append({
                    "id": row[0],
                    "content": content[:200] + "..." if len(content) > 200 else content,
                    "timestamp": row[2],
                    "session_name": row[3],
                    "model": row[4],
                    "session_id": row[5]
                })
            
            return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtém estatísticas globais"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total de sessões ativas
            cursor.execute('SELECT COUNT(*) FROM sessions WHERE is_active = 1')
            total_sessions = cursor.fetchone()[0]
            
            # Total de mensagens
            cursor.execute('SELECT COUNT(*) FROM messages')
            total_messages = cursor.fetchone()[0]
            
            # Total de snippets
            cursor.execute('SELECT COUNT(*) FROM code_snippets')
            total_snippets = cursor.fetchone()[0]
            
            # Total de arquivos
            cursor.execute('SELECT COUNT(*) FROM files')
            total_files = cursor.fetchone()[0]
            
            # Mensagens por modelo
            cursor.execute('''
                SELECT model, COUNT(*) as count
                FROM messages
                GROUP BY model
            ''')
            
            messages_by_model = {}
            for row in cursor.fetchall():
                messages_by_model[row[0]] = row[1]
            
            # Sessões por modelo
            cursor.execute('''
                SELECT model, COUNT(*) as count
                FROM sessions
                WHERE is_active = 1
                GROUP BY model
            ''')
            
            sessions_by_model = {}
            for row in cursor.fetchall():
                sessions_by_model[row[0]] = row[1]
            
            return {
                "total_sessions": total_sessions,
                "total_messages": total_messages,
                "total_code_snippets": total_snippets,
                "total_files": total_files,
                "messages_by_model": messages_by_model,
                "sessions_by_model": sessions_by_model
            }
    
    # Novos métodos para as melhorias
    def save_prompt_template(self, name: str, prompt: str, description: str = "", 
                            category: str = "general", is_public: bool = False) -> str:
        """Salva um template de prompt"""
        template_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO prompt_templates (id, name, description, prompt, category, is_public, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (template_id, name, description, prompt, category, int(is_public), datetime.now()))
            conn.commit()
        return template_id
    
    def get_prompt_templates(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Obtém templates de prompts"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            query = '''
                SELECT id, name, description, prompt, category, is_public, created_at, usage_count
                FROM prompt_templates
            '''
            if category:
                query += ' WHERE category = ?'
                cursor.execute(query, (category,))
            else:
                cursor.execute(query)
            
            templates = []
            for row in cursor.fetchall():
                templates.append({
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "prompt": row[3],
                    "category": row[4],
                    "is_public": bool(row[5]),
                    "created_at": row[6],
                    "usage_count": row[7]
                })
            return templates
    
    def increment_template_usage(self, template_id: str):
        """Incrementa o contador de uso de um template"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE prompt_templates SET usage_count = usage_count + 1 WHERE id = ?
            ''', (template_id,))
            conn.commit()
    
    def toggle_message_favorite(self, message_id: int, session_id: str, user_note: str = "") -> bool:
        """Adiciona/remove mensagem dos favoritos"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Verifica se já existe
            cursor.execute('''
                SELECT id FROM message_favorites WHERE message_id = ? AND session_id = ?
            ''', (message_id, session_id))
            
            exists = cursor.fetchone()
            
            if exists:
                # Remove dos favoritos
                cursor.execute('''
                    DELETE FROM message_favorites WHERE message_id = ? AND session_id = ?
                ''', (message_id, session_id))
                conn.commit()
                return False
            else:
                # Adiciona aos favoritos
                cursor.execute('''
                    INSERT INTO message_favorites (message_id, session_id, user_note, created_at)
                    VALUES (?, ?, ?, ?)
                ''', (message_id, session_id, user_note, datetime.now()))
                conn.commit()
                return True
    
    def get_favorite_messages(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Obtém mensagens favoritas"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if session_id:
                query = '''
                    SELECT mf.message_id, mf.user_note, mf.created_at, 
                           m.content, m.role, m.timestamp, s.name
                    FROM message_favorites mf
                    JOIN messages m ON mf.message_id = m.id
                    JOIN sessions s ON mf.session_id = s.id
                    WHERE mf.session_id = ?
                    ORDER BY mf.created_at DESC
                '''
                cursor.execute(query, (session_id,))
            else:
                query = '''
                    SELECT mf.message_id, mf.user_note, mf.created_at, 
                           m.content, m.role, m.timestamp, s.name, mf.session_id
                    FROM message_favorites mf
                    JOIN messages m ON mf.message_id = m.id
                    JOIN sessions s ON mf.session_id = s.id
                    ORDER BY mf.created_at DESC
                '''
                cursor.execute(query)
            
            favorites = []
            for row in cursor.fetchall():
                content = html.unescape(row[3])
                favorite = {
                    "message_id": row[0],
                    "user_note": row[1],
                    "favorited_at": row[2],
                    "content": content[:200] + "..." if len(content) > 200 else content,
                    "role": row[4],
                    "message_timestamp": row[5],
                    "session_name": row[6]
                }
                if not session_id:
                    favorite["session_id"] = row[7]
                favorites.append(favorite)
            
            return favorites
    
    def add_session_tags(self, session_id: str, tags: List[str], colors: Optional[Dict[str, str]] = None):
        """Adiciona tags a uma sessão"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for tag in tags:
                color = colors.get(tag, '#3b82f6') if colors else '#3b82f6'
                try:
                    cursor.execute('''
                        INSERT INTO session_tags (session_id, tag, color, created_at)
                        VALUES (?, ?, ?, ?)
                    ''', (session_id, tag, color, datetime.now()))
                except sqlite3.IntegrityError:
                    # Tag já existe para esta sessão
                    pass
            
            conn.commit()
    
    def remove_session_tag(self, session_id: str, tag: str):
        """Remove uma tag de uma sessão"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM session_tags WHERE session_id = ? AND tag = ?
            ''', (session_id, tag))
            conn.commit()
    
    def get_session_tags(self, session_id: str) -> List[Dict[str, str]]:
        """Obtém tags de uma sessão"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT tag, color FROM session_tags WHERE session_id = ? ORDER BY tag
            ''', (session_id,))
            
            return [{"tag": row[0], "color": row[1]} for row in cursor.fetchall()]
    
    def get_all_tags(self) -> List[Dict[str, Any]]:
        """Obtém todas as tags únicas com contagem"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT tag, color, COUNT(*) as count 
                FROM session_tags 
                GROUP BY tag, color 
                ORDER BY count DESC, tag
            ''')
            
            return [{"tag": row[0], "color": row[1], "count": row[2]} for row in cursor.fetchall()]
    
    def save_user_preference(self, key: str, value: str):
        """Salva preferência do usuário"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO user_preferences (key, value, updated_at)
                VALUES (?, ?, ?)
            ''', (key, value, datetime.now()))
            conn.commit()
    
    def get_user_preference(self, key: str, default: str = None) -> Optional[str]:
        """Obtém preferência do usuário"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT value FROM user_preferences WHERE key = ?', (key,))
            row = cursor.fetchone()
            return row[0] if row else default
    
    def get_sessions_by_tag(self, tag: str) -> List[Session]:
        """Obtém sessões por tag"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT s.id, s.name, s.created_at, s.last_accessed, s.model, s.message_count
                FROM sessions s
                JOIN session_tags st ON s.id = st.session_id
                WHERE st.tag = ? AND s.is_active = 1
                ORDER BY s.last_accessed DESC
            ''', (tag,))
            
            sessions = []
            for row in cursor.fetchall():
                sessions.append(Session(
                    id=row[0],
                    name=row[1],
                    created_at=datetime.fromisoformat(row[2]),
                    last_accessed=datetime.fromisoformat(row[3]),
                    model=AIModel[row[4]],
                    message_count=row[5]
                ))
            
            return sessions

class CodeExtractor:
    """Extrai e separa código do texto"""
    
    @staticmethod
    def extract_code_blocks(content: str) -> Tuple[str, List[Tuple[str, str, int]], List[str]]:
        """Extrai blocos de código e retorna texto limpo + lista de códigos"""
        code_pattern = r'```(\w+)?\n(.*?)```'
        snippets = []
        position = 0
        
        def replacer(match):
            nonlocal position
            language = match.group(1) or "text"
            code = match.group(2).strip()
            snippets.append((language, code, position))
            position += 1
            return f"[CÓDIGO {position}: {language}]"
        
        # Substituir blocos de código por placeholders
        clean_text = re.sub(code_pattern, replacer, content, flags=re.DOTALL)
        
        # Também extrair código inline
        inline_pattern = r'`([^`]+)`'
        inline_codes = re.findall(inline_pattern, clean_text)
        
        return clean_text, snippets, inline_codes
    
    @staticmethod
    def reconstruct_content(clean_text: str, snippets: List[Dict[str, Any]]) -> str:
        """Reconstrói o conteúdo original com os blocos de código"""
        content = clean_text
        for snippet in snippets:
            placeholder = f"[CÓDIGO {snippet['position'] + 1}: {snippet['language']}]"
            code_block = f"```{snippet['language']}\n{snippet['code']}\n```"
            content = content.replace(placeholder, code_block)
        return content

# ===== CLIENTES DE IA =====

class AIClient:
    """Cliente base para APIs de IA"""
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def send_message_stream(self, message: str, history: List[Dict[str, str]] = None) -> AsyncGenerator[str, None]:
        """Envia mensagem e retorna resposta em streaming"""
        raise NotImplementedError

class GeminiClient(AIClient):
    """Cliente para Gemini API com suporte a streaming"""
    def __init__(self):
        super().__init__(os.getenv('GEMINI_API_KEY'))
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-pro",
            generation_config={
                "max_output_tokens": 8192,
                "temperature": 0.7,
                "top_p": 0.95,
            }
        )
    
    async def send_message_stream(self, message: str, history: List[Dict[str, str]] = None) -> AsyncGenerator[str, None]:
        try:
            # Converter histórico para formato Gemini
            chat_history = []
            if history:
                for msg in history[:-1]:  # Excluir a última mensagem (atual)
                    if msg["role"] == "user":
                        chat_history.append({"role": "user", "parts": [msg["content"]]})
                    else:
                        chat_history.append({"role": "model", "parts": [msg["content"]]})
            
            # Criar chat com histórico
            chat = self.model.start_chat(history=chat_history)
            
            # Enviar mensagem com streaming
            response = await asyncio.to_thread(
                chat.send_message,
                message,
                stream=True
            )
            
            # Yield cada chunk
            for chunk in response:
                if chunk.text:
                    yield chunk.text
            
        except Exception as e:
            yield f"Erro ao comunicar com Gemini: {str(e)}"

class DeepSeekClient(AIClient):
    """Cliente para DeepSeek API com suporte a streaming"""
    def __init__(self):
        super().__init__(os.getenv('DEEPSEEK-API'))
        self.base_url = "https://api.deepseek.com/v1"
        self.model = "deepseek-reasoner"
    
    async def send_message_stream(self, message: str, history: List[Dict[str, str]] = None) -> AsyncGenerator[str, None]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Preparar mensagens com histórico
        messages = []
        if history:
            for msg in history[:-1]:
                role = "user" if msg["role"] == "user" else "assistant"
                messages.append({"role": role, "content": msg["content"]})
        
        messages.append({"role": "user", "content": message})
        
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 32768,
            "temperature": 0.7,
            "stream": True
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data
                ) as response:
                    if response.status == 200:
                        # Buffer para acumular resposta do reasoner
                        full_content = ""
                        in_answer_tag = False
                        answer_content = ""
                        
                        async for line in response.content:
                            line = line.decode('utf-8').strip()
                            if line.startswith("data: "):
                                line = line[6:]
                                if line == "[DONE]":
                                    if answer_content:
                                        yield answer_content
                                    break
                                
                                try:
                                    chunk_data = json.loads(line)
                                    if 'choices' in chunk_data and chunk_data['choices']:
                                        delta = chunk_data['choices'][0].get('delta', {})
                                        content = delta.get('content', '')
                                        
                                        if content:
                                            full_content += content
                                            
                                            # Para DeepSeek Reasoner, detectar tags <answer>
                                            if "<answer>" in content:
                                                in_answer_tag = True
                                                content = content.replace("<answer>", "")
                                            
                                            if "</answer>" in content:
                                                in_answer_tag = False
                                                content = content.replace("</answer>", "")
                                            
                                            # Se estamos dentro da tag answer, acumular
                                            if in_answer_tag:
                                                answer_content += content
                                            # Se não há tags answer, mostrar tudo
                                            elif "<answer>" not in full_content:
                                                yield content
                                
                                except json.JSONDecodeError:
                                    continue
                    else:
                        error_text = await response.text()
                        yield f"Erro DeepSeek API: {response.status} - {error_text}"
                        
        except Exception as e:
            yield f"Erro ao comunicar com DeepSeek: {str(e)}"

class GrokClient(AIClient):
    """Cliente para Grok API com suporte a streaming"""
    def __init__(self):
        super().__init__(os.getenv('GROK-API'))
        self.base_url = "https://api.x.ai/v1"
        self.model = "grok-3-mini"
    
    async def send_message_stream(self, message: str, history: List[Dict[str, str]] = None) -> AsyncGenerator[str, None]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Preparar mensagens com histórico
        messages = []
        if history:
            for msg in history[:-1]:
                role = "user" if msg["role"] == "user" else "assistant"
                messages.append({"role": role, "content": msg["content"]})
        
        messages.append({"role": "user", "content": message})
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096,
            "stream": True
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data
                ) as response:
                    if response.status == 200:
                        async for line in response.content:
                            line = line.decode('utf-8').strip()
                            if line.startswith("data: "):
                                line = line[6:]
                                if line == "[DONE]":
                                    break
                                
                                try:
                                    chunk_data = json.loads(line)
                                    if 'choices' in chunk_data and chunk_data['choices']:
                                        delta = chunk_data['choices'][0].get('delta', {})
                                        content = delta.get('content', '')
                                        if content:
                                            yield content
                                except json.JSONDecodeError:
                                    continue
                    else:
                        error_text = await response.text()
                        yield f"Erro Grok API: {response.status} - {error_text}"
                        
        except Exception as e:
            yield f"Erro ao comunicar com Grok: {str(e)}"

# ===== MODELOS PYDANTIC =====

class SessionCreate(BaseModel):
    name: str
    model: str

class MessageRequest(BaseModel):
    session_id: str
    message: str
    model: str

class ModelUpdate(BaseModel):
    model: str

class SessionUpdate(BaseModel):
    name: Optional[str] = None
    model: Optional[str] = None

class PromptTemplateCreate(BaseModel):
    name: str
    description: str = ""
    prompt: str
    category: str = "general"
    is_public: bool = False

class MessageFavoriteToggle(BaseModel):
    message_id: int
    user_note: str = ""

class SessionTagsUpdate(BaseModel):
    tags: List[str]
    colors: Optional[Dict[str, str]] = None

class UserPreferenceUpdate(BaseModel):
    key: str
    value: str

# ===== APLICAÇÃO FASTAPI =====

app = FastAPI(
    title="Multi-AI Chatbot API",
    description="API para chatbot com múltiplas IAs",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Criar diretórios necessários
Path("templates").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)
Path("uploads").mkdir(exist_ok=True)

# Servir arquivos estáticos
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Inicializar componentes
db = DatabaseManager()
clients = {
    AIModel.GEMINI: GeminiClient(),
    AIModel.DEEPSEEK: DeepSeekClient(),
    AIModel.GROK: GrokClient()
}

# WebSocket connections
websocket_connections: Dict[str, WebSocket] = {}

# ===== ROTAS DA API =====

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Servir o frontend SPA"""
    template_path = Path("templates/index.html")
    if template_path.exists():
        return HTMLResponse(content=template_path.read_text(encoding="utf-8"))
    else:
        return HTMLResponse(content="<h1>Multi-AI Chatbot API</h1><p>Frontend não encontrado. Coloque o arquivo index.html em templates/</p>")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# ===== SESSÕES =====

@app.get("/api/sessions")
async def get_sessions(active_only: bool = True, tag: Optional[str] = None):
    """Listar todas as sessões com tags"""
    if tag:
        sessions = db.get_sessions_by_tag(tag)
    else:
        sessions = db.get_sessions(active_only)
    
    sessions_with_tags = []
    for s in sessions:
        session_dict = {
            "id": s.id,
            "name": s.name,
            "model": s.model.value,
            "createdAt": s.created_at.isoformat(),
            "lastAccessed": s.last_accessed.isoformat(),
            "messageCount": s.message_count,
            "tags": db.get_session_tags(s.id)
        }
        sessions_with_tags.append(session_dict)
    
    return sessions_with_tags

@app.post("/api/sessions")
async def create_session(session_data: SessionCreate):
    """Criar nova sessão"""
    try:
        model_mapping = {
            'Gemini 2.5 Pro': AIModel.GEMINI,
            'DeepSeek Reasoner': AIModel.DEEPSEEK,
            'Grok 3 Mini': AIModel.GROK
        }
        
        model = model_mapping.get(session_data.model, AIModel.GEMINI)
        logger.info(f"Criando sessão com modelo: {session_data.model} -> {model}")
        
    except Exception as e:
        logger.error(f"Erro ao mapear modelo: {e}")
        model = AIModel.GEMINI
    
    session = db.create_session(session_data.name, model)
    
    return {
        "id": session.id,
        "name": session.name,
        "model": session.model.value,
        "createdAt": session.created_at.isoformat(),
        "lastAccessed": session.last_accessed.isoformat(),
        "messageCount": 0,
        "tags": []
    }

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Obter sessão e suas mensagens"""
    session = db.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    
    messages = db.get_session_messages(session_id)
    
    # Formatar mensagens para o frontend
    formatted_messages = []
    for msg in messages:
        formatted_msg = {
            "id": msg["id"],
            "role": msg["role"],
            "content": msg["content"],
            "timestamp": msg["timestamp"],
            "model": msg["model"],
            "codeSnippets": msg["code_snippets"],
            "is_favorite": msg["is_favorite"],
            "favorite_note": msg["favorite_note"]
        }
        formatted_messages.append(formatted_msg)
    
    return {
        "session": {
            "id": session.id,
            "name": session.name,
            "model": session.model.value,
            "createdAt": session.created_at.isoformat(),
            "lastAccessed": session.last_accessed.isoformat(),
            "messageCount": session.message_count,
            "tags": db.get_session_tags(session.id)
        },
        "messages": formatted_messages
    }

@app.put("/api/sessions/{session_id}")
async def update_session(session_id: str, update_data: SessionUpdate):
    """Atualizar sessão"""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    
    # Atualizar nome se fornecido
    if update_data.name:
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE sessions SET name = ? WHERE id = ?', 
                         (update_data.name, session_id))
            conn.commit()
    
    # Atualizar modelo se fornecido
    if update_data.model:
        try:
            model_mapping = {
                'Gemini 2.5 Pro': AIModel.GEMINI,
                'DeepSeek Reasoner': AIModel.DEEPSEEK,
                'Grok 3 Mini': AIModel.GROK
            }
            model = model_mapping.get(update_data.model)
            if model:
                db.update_session_model(session_id, model)
            else:
                raise HTTPException(status_code=400, detail="Modelo inválido")
        except Exception as e:
            logger.error(f"Erro ao atualizar modelo: {e}")
            raise HTTPException(status_code=400, detail="Erro ao atualizar modelo")
    
    return {"message": "Sessão atualizada com sucesso"}

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Deletar sessão (soft delete)"""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    
    db.soft_delete_session(session_id)
    return {"message": "Sessão deletada com sucesso"}

@app.put("/api/sessions/{session_id}/model")
async def update_session_model(session_id: str, model_update: ModelUpdate):
    """Atualizar modelo da sessão"""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    
    try:
        model_mapping = {
            'Gemini 2.5 Pro': AIModel.GEMINI,
            'DeepSeek Reasoner': AIModel.DEEPSEEK,
            'Grok 3 Mini': AIModel.GROK
        }
        model = model_mapping.get(model_update.model)
        if model:
            db.update_session_model(session_id, model)
            logger.info(f"Modelo da sessão {session_id} atualizado para: {model}")
        else:
            raise HTTPException(status_code=400, detail="Modelo inválido")
    except Exception as e:
        logger.error(f"Erro ao atualizar modelo: {e}")
        raise HTTPException(status_code=400, detail="Erro ao atualizar modelo")
    
    return {"message": "Modelo atualizado com sucesso"}

# ===== TEMPLATES =====

@app.post("/api/templates")
async def create_prompt_template(template: PromptTemplateCreate):
    """Cria um novo template de prompt"""
    template_id = db.save_prompt_template(
        name=template.name,
        prompt=template.prompt,
        description=template.description,
        category=template.category,
        is_public=template.is_public
    )
    
    return {
        "id": template_id,
        "message": "Template criado com sucesso"
    }

@app.get("/api/templates")
async def get_templates(category: Optional[str] = None):
    """Lista templates de prompts"""
    templates = db.get_prompt_templates(category)
    return {
        "templates": templates,
        "total": len(templates)
    }

@app.post("/api/templates/{template_id}/use")
async def use_template(template_id: str):
    """Registra uso de um template"""
    db.increment_template_usage(template_id)
    return {"message": "Uso registrado"}

# ===== FAVORITOS =====

@app.post("/api/sessions/{session_id}/favorites")
async def toggle_favorite(session_id: str, data: MessageFavoriteToggle):
    """Adiciona/remove mensagem dos favoritos"""
    is_favorited = db.toggle_message_favorite(
        data.message_id,
        session_id,
        data.user_note
    )
    
    return {
        "favorited": is_favorited,
        "message": "Favorito adicionado" if is_favorited else "Favorito removido"
    }

@app.get("/api/favorites")
async def get_favorites(session_id: Optional[str] = None):
    """Lista mensagens favoritas"""
    favorites = db.get_favorite_messages(session_id)
    return {
        "favorites": favorites,
        "total": len(favorites)
    }

# ===== TAGS =====

@app.put("/api/sessions/{session_id}/tags")
async def update_session_tags(session_id: str, data: SessionTagsUpdate):
    """Atualiza tags de uma sessão"""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    
    # Remove tags antigas
    current_tags = db.get_session_tags(session_id)
    for tag_info in current_tags:
        db.remove_session_tag(session_id, tag_info["tag"])
    
    # Adiciona novas tags
    db.add_session_tags(session_id, data.tags, data.colors)
    
    return {"message": "Tags atualizadas com sucesso"}

@app.get("/api/sessions/{session_id}/tags")
async def get_session_tags(session_id: str):
    """Obtém tags de uma sessão"""
    tags = db.get_session_tags(session_id)
    return {"tags": tags}

@app.get("/api/tags")
async def get_all_tags():
    """Lista todas as tags com estatísticas"""
    tags = db.get_all_tags()
    return {"tags": tags}

@app.get("/api/sessions/by-tag/{tag}")
async def get_sessions_by_tag(tag: str):
    """Lista sessões por tag"""
    sessions = db.get_sessions_by_tag(tag)
    return [
        {
            "id": s.id,
            "name": s.name,
            "model": s.model.value,
            "createdAt": s.created_at.isoformat(),
            "lastAccessed": s.last_accessed.isoformat(),
            "messageCount": s.message_count,
            "tags": db.get_session_tags(s.id)
        }
        for s in sessions
    ]

# ===== PREFERÊNCIAS =====

@app.put("/api/preferences")
async def update_preference(pref: UserPreferenceUpdate):
    """Atualiza preferência do usuário"""
    db.save_user_preference(pref.key, pref.value)
    return {"message": "Preferência salva"}

@app.get("/api/preferences/{key}")
async def get_preference(key: str, default: Optional[str] = None):
    """Obtém preferência do usuário"""
    value = db.get_user_preference(key, default)
    return {"key": key, "value": value}

# ===== MENSAGENS E BUSCA =====

@app.get("/api/search")
async def search_messages(q: str, session_id: Optional[str] = None):
    """Buscar mensagens"""
    if not q:
        return []
    
    results = db.search_messages(q, session_id)
    
    formatted_results = []
    for r in results:
        formatted_results.append({
            "id": str(r["id"]),
            "sessionId": r["session_id"],
            "sessionName": r["session_name"],
            "model": r["model"],
            "timestamp": r["timestamp"],
            "preview": r["content"]
        })
    
    return formatted_results

@app.get("/api/stats")
async def get_stats():
    """Obter estatísticas"""
    stats = db.get_statistics()
    
    return {
        "totalSessions": stats["total_sessions"],
        "totalMessages": stats["total_messages"],
        "totalCodeSnippets": stats["total_code_snippets"],
        "totalFiles": stats["total_files"],
        "messagesByModel": stats["messages_by_model"],
        "sessionsByModel": stats["sessions_by_model"]
    }

# ===== UPLOAD DE ARQUIVOS =====

@app.post("/api/upload/{session_id}")
async def upload_file(session_id: str, file: UploadFile = File(...)):
    """Upload de arquivo para uma sessão"""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    
    # Salvar arquivo
    file_path = Path("uploads") / session_id / file.filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    # Salvar no banco
    db.save_file(session_id, file.filename, file.content_type, str(file_path))
    
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(content),
        "path": str(file_path)
    }

@app.get("/api/export/{session_id}")
async def export_session(session_id: str):
    """Exportar sessão como JSON"""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    
    messages = db.get_session_messages(session_id)
    
    export_data = {
        "session": {
            "id": session.id,
            "name": session.name,
            "model": session.model.value,
            "created_at": session.created_at.isoformat(),
            "last_accessed": session.last_accessed.isoformat(),
            "message_count": session.message_count,
            "tags": db.get_session_tags(session.id)
        },
        "messages": messages,
        "exported_at": datetime.now().isoformat()
    }
    
    # Criar arquivo temporário
    export_path = Path("exports") / f"{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    export_path.parent.mkdir(exist_ok=True)
    
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    return FileResponse(
        path=str(export_path),
        filename=f"{session.name.replace(' ', '_')}_export.json",
        media_type="application/json"
    )

# ===== WEBSOCKET =====

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint para streaming de mensagens"""
    await websocket.accept()
    websocket_id = str(uuid.uuid4())
    websocket_connections[websocket_id] = websocket
    
    try:
        while True:
            # Receber mensagem
            data = await websocket.receive_json()
            
            if data["type"] == "message":
                await handle_message(websocket, data)
            elif data["type"] == "ping":
                await websocket.send_json({"type": "pong"})
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket {websocket_id} desconectado")
    except Exception as e:
        logger.error(f"Erro no WebSocket {websocket_id}: {e}")
    finally:
        if websocket_id in websocket_connections:
            del websocket_connections[websocket_id]

async def handle_message(websocket: WebSocket, data: Dict[str, Any]):
    """Processar mensagem recebida via WebSocket"""
    session_id = data.get("sessionId")
    message = data.get("message")
    model_name = data.get("model")
    
    if not all([session_id, message, model_name]):
        await websocket.send_json({
            "type": "error",
            "message": "Dados incompletos"
        })
        return
    
    # Verificar sessão
    session = db.get_session(session_id)
    if not session:
        await websocket.send_json({
            "type": "error",
            "message": "Sessão não encontrada"
        })
        return
    
    try:
        # Mapeamento correto dos modelos
        model_mapping = {
            'Gemini 2.5 Pro': AIModel.GEMINI,
            'DeepSeek Reasoner': AIModel.DEEPSEEK,
            'Grok 3 Mini': AIModel.GROK
        }
        
        # Usar o modelo especificado ou o modelo da sessão como fallback
        model = model_mapping.get(model_name, session.model)
        logger.info(f"Usando modelo: {model_name} -> {model}")
        
    except Exception as e:
        logger.error(f"Erro ao mapear modelo: {e}")
        model = session.model
    
    # Salvar mensagem do usuário
    db.save_message(session_id, "user", message, model)
    db.update_session_access(session_id)
    
    # Obter histórico
    history = db.get_session_messages(session_id)
    
    # Converter para formato dos clientes
    client_history = []
    for msg in history:
        content = msg["content"]
        if msg["code_snippets"]:
            content = CodeExtractor.reconstruct_content(content, msg["code_snippets"])
        client_history.append({"role": msg["role"], "content": content})
    
    # Obter cliente correto
    client = clients[model]
    
    # Streaming da resposta
    full_response = ""
    
    try:
        # Enviar início do streaming
        await websocket.send_json({
            "type": "stream_start",
            "model": model.value
        })
        
        # Stream chunks
        async for chunk in client.send_message_stream(message, client_history):
            full_response += chunk
            
            # Enviar chunk via WebSocket
            await websocket.send_json({
                "type": "stream",
                "chunk": chunk
            })
            
            await asyncio.sleep(0.01)  # Pequeno delay para suavizar
        
        # Extrair código
        clean_text, code_snippets, _ = CodeExtractor.extract_code_blocks(full_response)
        
        # Salvar resposta
        message_id = db.save_message(
            session_id,
            "assistant",
            clean_text,
            model,
            len(full_response.split())
        )
        
        # Salvar snippets
        if code_snippets:
            db.save_code_snippets(message_id, code_snippets)
        
        # Enviar conclusão
        await websocket.send_json({
            "type": "message_complete",
            "fullResponse": full_response,
            "codeSnippets": [
                {
                    "language": lang,
                    "code": code,
                    "position": pos
                }
                for lang, code, pos in code_snippets
            ]
        })
        
    except Exception as e:
        logger.error(f"Erro ao processar mensagem: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"Erro ao processar mensagem: {str(e)}"
        })

# ===== INICIALIZAÇÃO =====

@app.on_event("startup")
async def startup_event():
    """Executado na inicialização"""
    logger.info("Multi-AI Chatbot API iniciada")
    logger.info(f"Banco de dados: {Path(db.db_path).absolute()}")
    
    # Verificar API keys
    required_keys = ['GEMINI_API_KEY', 'DEEPSEEK-API', 'GROK-API']
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        logger.warning(f"API keys ausentes: {missing_keys}")
    else:
        logger.info("Todas as API keys configuradas")

@app.on_event("shutdown")
async def shutdown_event():
    """Executado no encerramento"""
    logger.info("Encerrando Multi-AI Chatbot API")
    
    # Fechar conexões WebSocket
    for ws in websocket_connections.values():
        await ws.close()

# ===== EXECUÇÃO =====

if __name__ == "__main__":
    import uvicorn
    
    # Configurações do servidor
    config = {
        "host": "0.0.0.0",
        "port": 8000,
        "reload": True,
        "log_level": "info"
    }
    
    logger.info(f"Iniciando servidor em http://{config['host']}:{config['port']}")
    logger.info("Documentação da API: http://localhost:8000/docs")
    
    uvicorn.run("app:app", **config)
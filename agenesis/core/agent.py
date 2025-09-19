from typing import Optional, Dict, Any, Union
from pathlib import Path

from ..perception import TextPerception, PerceptionResult
from ..memory import ImmediateMemory, WorkingMemory, FileMemory, SQLiteMemory
from ..cognition import BasicCognition
from ..action import BasicAction
from ..evolution import EvolutionAnalyzer
from ..persona import PersonaContext, BasePersona, load_persona


class Agent:
    """Main Agent class with profile-based memory management"""
    
    def __init__(self, profile: Optional[str] = None, config: Optional[Dict[str, Any]] = None, 
                 persona: Optional[Union[str, Dict[str, Any]]] = None, persona_config: Optional[str] = None):
        self.profile = profile
        self.config = config or {}
        
        # Initialize perception
        self.perception = TextPerception(self.config.get('perception', {}))
        
        # Initialize memory based on profile
        self._init_memory()
        
        # Initialize cognition, action, and evolution
        self.cognition = BasicCognition(self.config.get('cognition', {}))
        self.action = BasicAction(self.config.get('action', {}))
        self.evolution = EvolutionAnalyzer(self.config.get('evolution', {}))
        
        # Initialize persona
        self.persona = self._init_persona(persona, persona_config)
    
    def _init_memory(self):
        """Initialize memory systems based on profile"""
        # Always have immediate and working memory
        self.immediate_memory = ImmediateMemory(self.config.get('immediate_memory', {}))
        self.working_memory = WorkingMemory(self.config.get('working_memory', {}))
        
        # Persistent memory only for named profiles
        if self.profile:
            self._init_persistent_memory()
            self.has_persistent_memory = True
        else:
            self.persistent_memory = None
            self.has_persistent_memory = False
    
    def _init_persistent_memory(self):
        """Initialize persistent memory for named profiles"""
        profile_dir = Path.home() / '.agenesis' / 'profiles' / self.profile
        profile_dir.mkdir(parents=True, exist_ok=True)
        
        # Choose storage type from config (default to FileMemory)
        storage_type = self.config.get('storage_type', 'file')
        
        if storage_type == 'sqlite':
            db_path = profile_dir / 'memory.db'
            self.persistent_memory = SQLiteMemory({'db_path': str(db_path)})
        else:  # file storage
            self.persistent_memory = FileMemory({'storage_dir': str(profile_dir)})
    
    def _init_persona(self, persona: Optional[Union[str, Dict[str, Any]]], 
                     persona_config: Optional[str]) -> Optional[BasePersona]:
        """Initialize persona from various sources"""
        if persona is None and persona_config is None:
            return None
        
        # Determine source for persona loading
        source = persona if persona is not None else persona_config
        
        try:
            return load_persona(source)
        except Exception as e:
            print(f"Warning: Failed to load persona: {e}")
            return None
    
    def set_persona(self, persona: Optional[Union[str, Dict[str, Any]]]):
        """Set or change persona at runtime"""
        self.persona = self._init_persona(persona, None)
    
    async def process_input(self, text_input: str) -> str:
        """Main processing pipeline: perception → memory → cognition → action"""
        # 1. Generate persona context (if persona available)
        persona_context = self.persona.create_context(text_input) if self.persona else None
        
        # 2. Perceive input (with optional persona context)
        perception_result = self.perception.process(text_input, persona_context)
        
        # 3. Store in memory systems (with optional persona context)
        memory_context = {"persona_context": persona_context.to_dict()} if persona_context else None
        self.immediate_memory.store(perception_result, memory_context)
        self.working_memory.store(perception_result, memory_context)
        if self.has_persistent_memory:
            self.persistent_memory.store(perception_result, memory_context)
        
        # 4. Cognitive processing (with optional persona context)
        cognition_result = await self.cognition.process(self.immediate_memory, self.working_memory, persona_context)
        
        # 5. Action generation (with optional persona context)
        action_result = await self.action.generate_response(cognition_result, persona_context)
        
        # 6. Evolution analysis (if persistent memory available)
        if self.has_persistent_memory:
            evolution_decision = await self.evolution.analyze_memory_session(
                self.immediate_memory, self.working_memory
            )
            
            if evolution_decision.should_persist:
                print(f"🧠 Learning detected: {evolution_decision.learning_description}")
                
                # Create evolved knowledge metadata
                evolved_metadata = self.evolution.create_evolved_knowledge_metadata(evolution_decision)
                
                # Mark recent memories as evolved knowledge and store
                recent_memories = self.working_memory.get_recent(3)
                for memory in recent_memories:
                    memory.is_evolved_knowledge = True
                    memory.evolution_metadata = {
                        "knowledge_summary": evolved_metadata.knowledge_summary,
                        "learning_context": evolved_metadata.learning_context,
                        "future_relevance": evolved_metadata.future_relevance,
                        "evolved_at": evolved_metadata.evolved_at.isoformat()
                    }
                    memory.reliability_multiplier = evolved_metadata.reliability_boost
                    
                    # Store enhanced memory in persistent storage
                    self.persistent_memory.store(memory.perception_result, memory.context)
        
        # 7. Return the response text to user
        return action_result.response_text
    
    def get_current_focus(self) -> Optional[PerceptionResult]:
        """Get what the agent is currently focused on"""
        record = self.immediate_memory.get_current()
        return record.perception_result if record else None
    
    def get_session_context(self, count: int = 5) -> list:
        """Get recent session context"""
        records = self.working_memory.get_recent(count)
        return [record.perception_result for record in records]
    
    def clear_focus(self):
        """Clear current focus - ready for next input"""
        self.immediate_memory.clear()
    
    def end_session(self):
        """End current session, consolidate memory if needed"""
        # Clear working memory
        self.working_memory.clear()
        
        # Clear immediate memory
        self.immediate_memory.clear()
        
        # Persistent memory remains for next session
    
    def get_profile_info(self) -> Dict[str, Any]:
        """Get information about current agent profile"""
        return {
            'profile': self.profile,
            'is_anonymous': self.profile is None,
            'has_persistent_memory': self.has_persistent_memory,
            'storage_location': str(Path.home() / '.agenesis' / 'profiles' / self.profile) if self.profile else None,
            'current_focus': self.get_current_focus() is not None,
            'session_size': self.working_memory.size(),
            'persona': {
                'name': self.persona.get_name() if self.persona else None,
                'description': self.persona.get_description() if self.persona else None,
                'active': self.persona is not None
            }
        }
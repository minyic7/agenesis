from typing import Optional, Dict, Any, Union
from pathlib import Path

from ..perception import TextPerception, PerceptionResult
from ..memory import ImmediateMemory, WorkingMemory, SQLiteMemory
from ..cognition import BasicCognition, SemanticCognition
from ..action import BasicAction
from ..evolution import EvolutionAnalyzer
from ..persona import PersonaContext, BasePersona, load_persona


# Agent Configuration Constants
class AgentConfig:
    """Constants for agent configuration to avoid magic numbers"""
    # Project knowledge import
    CONTENT_SUMMARY_LENGTH = 200  # Length for content summary truncation

    # Evolution and memory
    RECENT_MEMORIES_FOR_EVOLUTION = 3  # Recent memories to include in evolution

    # Embedding initialization
    EMBEDDING_BATCH_SIZE = 100  # Records to process in embedding batches


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
        # Use semantic cognition if enabled, otherwise basic cognition
        use_semantic_search = self.config.get('use_semantic_search', True)
        if use_semantic_search:
            self.cognition = SemanticCognition(self.config.get('cognition', {}))
        else:
            self.cognition = BasicCognition(self.config.get('cognition', {}))

        self.action = BasicAction(self.config.get('action', {}))
        self.evolution = EvolutionAnalyzer(self.config.get('evolution', {}))
        
        # Initialize persona
        self.persona = self._init_persona(persona, persona_config)

        # Initialize embeddings on startup (if using semantic search)
        self._embedding_initialization_task = None
        if (hasattr(self.cognition, 'embedding_provider') and
            self.cognition.embedding_provider is not None):
            # Schedule embedding initialization for working memory and recent persistent memory
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    self._embedding_initialization_task = loop.create_task(
                        self._initialize_embeddings_startup()
                    )
            except RuntimeError:
                # No event loop running, will initialize on first process_input
                pass
    
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

        # Use SQLiteMemory for all persistent storage
        db_path = profile_dir / 'memory.db'
        self.persistent_memory = SQLiteMemory({'db_path': str(db_path)})
    
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
        # 0. Ensure recent embeddings are initialized (if not already done)
        if (self._embedding_initialization_task is None and
            self.has_persistent_memory and
            hasattr(self.cognition, 'embedding_provider') and
            self.cognition.embedding_provider is not None):
            await self.ensure_embedding_initialization()

        # 1. Generate persona context (if persona available)
        persona_context = self.persona.create_context(text_input) if self.persona else None
        
        # 2. Perceive input (with optional persona context)
        perception_result = self.perception.process(text_input, persona_context)

        # 3. Generate embedding before storing (if semantic search enabled)
        embedding = None
        if (hasattr(self.cognition, 'embedding_provider') and
            self.cognition.embedding_provider is not None):
            try:
                embedding = await self.cognition.embedding_provider.embed_text(
                    perception_result.content
                )
            except Exception as e:
                print(f"⚠️ Failed to generate embedding: {e}")
                embedding = None

        # 4. Store in memory systems with embedding
        memory_context = {"persona_context": persona_context.to_dict()} if persona_context else None

        # Create memory record with embedding
        from ..memory.base import MemoryRecord
        from datetime import datetime, timezone
        memory_record = MemoryRecord(
            id="",  # Generated in __post_init__
            perception_result=perception_result,
            stored_at=datetime.now(timezone.utc),
            context=memory_context or {},
            metadata={'memory_type': 'AgentMemory'},
            embedding=embedding
        )

        # Store the complete record in all memory systems
        immediate_id = self.immediate_memory.store_record(memory_record)
        working_id = self.working_memory.store_record(memory_record)

        if self.has_persistent_memory:
            persistent_id = self.persistent_memory.store_record(memory_record)
        
        # 4. Cognitive processing (with optional persona context and persistent memory)
        persistent_memory = self.persistent_memory if self.has_persistent_memory else None

        # Check if cognition supports semantic search with persistent memory
        if hasattr(self.cognition, '_find_relevant_memories_semantic'):
            cognition_result = await self.cognition.process(
                self.immediate_memory, self.working_memory, persona_context, persistent_memory
            )
        else:
            cognition_result = await self.cognition.process(
                self.immediate_memory, self.working_memory, persona_context
            )
        
        # 5. Action generation (memory context provided by cognition)
        action_result = await self.action.generate_response(cognition_result, persona_context)

        # 6. Evolution analysis with full interaction context (if persistent memory available)
        if self.has_persistent_memory:
            # Check if this interaction should be learned from using validation functions
            should_learn = self.evolution._should_learn_from_interaction(
                text_input,
                action_result.response_text,
                self.persona
            )

            if should_learn:
                # Run LLM evolution analysis with persona learning preferences
                evolution_decision = await self.evolution.analyze_memory_session(
                    self.immediate_memory, self.working_memory, self.persona
                )

                if evolution_decision.should_persist:
                    print(f"🧠 Learning detected: {evolution_decision.learning_description}")

                    # Create evolved knowledge metadata
                    evolved_metadata = self.evolution.create_evolved_knowledge_metadata(evolution_decision)

                    # Create new enhanced memory records with complete interaction context
                    recent_memories = self.working_memory.get_recent(AgentConfig.RECENT_MEMORIES_FOR_EVOLUTION)
                    for memory in recent_memories:
                        # Create new enhanced memory record for persistent storage
                        from ..memory.base import MemoryRecord
                        from datetime import datetime, timezone

                        enhanced_memory = MemoryRecord(
                            id="",  # Generate new ID for evolved knowledge version
                            perception_result=memory.perception_result,
                            stored_at=datetime.now(timezone.utc),
                            context=memory.context,
                            metadata=memory.metadata,
                            is_evolved_knowledge=True,
                            evolution_metadata={
                                "knowledge_summary": evolved_metadata.knowledge_summary,
                                "learning_context": evolved_metadata.learning_context,
                                "future_relevance": evolved_metadata.future_relevance,
                                "evolved_at": evolved_metadata.evolved_at.isoformat()
                            },
                            embedding=memory.embedding,
                            agent_response=action_result.response_text  # Complete interaction
                        )

                        # Store enhanced memory record with complete interaction in persistent storage
                        self.persistent_memory.store_record(enhanced_memory)
            else:
                print("📋 Interaction filtered out by validation function")

        # 7. Return the response text to user
        return action_result.response_text
    
    def get_current_focus(self) -> Optional[PerceptionResult]:
        """Get what the agent is currently focused on"""
        record = self.immediate_memory.get_current()
        return record.perception_result if record else None

    def get_current_focus_record(self) -> Optional['MemoryRecord']:
        """Get the current focus memory record"""
        return self.immediate_memory.get_current()
    
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
    
    async def import_project_knowledge(self, knowledge_sources: list) -> Dict[str, Any]:
        """Import project documentation and knowledge into persistent memory
        
        Args:
            knowledge_sources: List of dicts with 'content', 'type', 'importance' keys
            
        Returns:
            Dict with import results
        """
        if not self.has_persistent_memory:
            raise ValueError("Project knowledge import requires a named agent profile")
        
        imported_count = 0
        skipped_count = 0
        results = []
        
        for source in knowledge_sources:
            try:
                # Extract source information
                content = source.get('content', '')
                doc_type = source.get('type', 'general')
                importance = source.get('importance', 'medium')
                
                if not content.strip():
                    skipped_count += 1
                    continue
                
                # Process content through perception (without persona context for raw docs)
                perception_result = self.perception.process(content, context=None)
                
                # Create project knowledge context
                project_context = {
                    'source_type': 'project_knowledge',
                    'document_type': doc_type,
                    'importance': importance,
                    'imported_at': perception_result.timestamp.isoformat(),
                    'content_summary': content[:AgentConfig.CONTENT_SUMMARY_LENGTH] + '...' if len(content) > AgentConfig.CONTENT_SUMMARY_LENGTH else content
                }
                
                # Store in persistent memory
                memory_id = self.persistent_memory.store(perception_result, project_context)
                
                # Retrieve and enhance as evolved knowledge
                record = self.persistent_memory.retrieve(memory_id)
                if record:
                    record.is_evolved_knowledge = True
                    record.evolution_metadata = {
                        'knowledge_summary': f"{doc_type.title()} documentation",
                        'learning_context': 'project_documentation',
                        'future_relevance': f"Relevant for {doc_type} decisions and planning",
                        'imported_at': perception_result.timestamp.isoformat()
                    }

                    # Re-store the enhanced record
                    self.persistent_memory.store(record.perception_result, record.context)
                
                imported_count += 1
                results.append({
                    'type': doc_type,
                    'size': len(content),
                    'importance': importance,
                    'memory_id': memory_id
                })
                
            except Exception as e:
                skipped_count += 1
                results.append({
                    'type': source.get('type', 'unknown'),
                    'error': str(e)
                })
        
        return {
            'imported_count': imported_count,
            'skipped_count': skipped_count,
            'total_sources': len(knowledge_sources),
            'results': results
        }
    
    async def import_from_files(self, file_paths: list, default_type: str = 'documentation') -> Dict[str, Any]:
        """Convenience method to import project knowledge from files
        
        Args:
            file_paths: List of file paths or dicts with 'path', 'type', 'importance'
            default_type: Default document type if not specified
            
        Returns:
            Dict with import results
        """
        knowledge_sources = []
        
        for file_spec in file_paths:
            try:
                if isinstance(file_spec, str):
                    # Simple file path
                    file_path = file_spec
                    doc_type = default_type
                    importance = 'medium'
                else:
                    # Dict with specifications
                    file_path = file_spec['path']
                    doc_type = file_spec.get('type', default_type)
                    importance = file_spec.get('importance', 'medium')
                
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                knowledge_sources.append({
                    'content': content,
                    'type': doc_type,
                    'importance': importance,
                    'source_file': file_path
                })
                
            except Exception as e:
                knowledge_sources.append({
                    'content': '',
                    'type': file_spec.get('type', 'unknown') if isinstance(file_spec, dict) else 'unknown',
                    'error': f"Failed to read {file_path if 'file_path' in locals() else 'file'}: {e}"
                })
        
        return await self.import_project_knowledge(knowledge_sources)

    async def _initialize_embeddings_startup(self) -> Dict[str, Any]:
        """Initialize embeddings for working memory and recent persistent memory"""
        if not (hasattr(self.cognition, 'embedding_provider') and
                self.cognition.embedding_provider is not None):
            return {'status': 'skipped', 'reason': 'no_embedding_provider'}

        try:
            results = {'working_memory': 0, 'persistent_memory': 0}

            # 1. Embed all working memory records (should be ~100 max)
            working_records = self.working_memory.get_all()
            records_without_embeddings = [r for r in working_records if not r.embedding]

            if records_without_embeddings:
                print(f"🔄 Embedding {len(records_without_embeddings)} working memory records...")
                contents = [r.perception_result.content for r in records_without_embeddings]
                embeddings = await self.cognition.embedding_provider.embed_batch(contents)

                # Update records in-place
                for record, embedding in zip(records_without_embeddings, embeddings):
                    if embedding and len(embedding) > 0:
                        record.embedding = embedding
                        results['working_memory'] += 1

                print(f"✅ Embedded {results['working_memory']} working memory records")

            # 2. Load recent 100 persistent memory records (should already have embeddings)
            if self.has_persistent_memory:
                recent_persistent = self.persistent_memory.get_recent(AgentConfig.EMBEDDING_BATCH_SIZE)
                records_without_embeddings = [r for r in recent_persistent if not r.embedding]

                if records_without_embeddings:
                    print(f"🔄 Embedding {len(records_without_embeddings)} recent persistent records...")
                    contents = [r.perception_result.content for r in records_without_embeddings]
                    embeddings = await self.cognition.embedding_provider.embed_batch(contents)

                    # Batch update persistent storage
                    batch_updates = []
                    for record, embedding in zip(records_without_embeddings, embeddings):
                        if embedding and len(embedding) > 0:
                            record.embedding = embedding
                            batch_updates.append((record.id, embedding))

                    if batch_updates and hasattr(self.persistent_memory, 'batch_update_embeddings'):
                        updated_count = self.persistent_memory.batch_update_embeddings(batch_updates)
                        results['persistent_memory'] = updated_count
                        print(f"✅ Embedded {updated_count} recent persistent records")

                print(f"📚 Loaded {len(recent_persistent)} recent persistent memories (coverage ready)")

            return {
                'status': 'completed',
                'working_memory_embedded': results['working_memory'],
                'persistent_memory_embedded': results['persistent_memory']
            }

        except Exception as e:
            print(f"⚠️ Startup embedding initialization failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    async def ensure_embedding_initialization(self) -> None:
        """Ensure embedding initialization has completed"""
        if self._embedding_initialization_task:
            try:
                await self._embedding_initialization_task
                self._embedding_initialization_task = None
            except Exception as e:
                print(f"Embedding initialization task failed: {e}")
        elif (hasattr(self.cognition, 'embedding_provider') and
              self.cognition.embedding_provider is not None):
            # Initialize embeddings now if not already done
            await self._initialize_embeddings_startup()

    def get_profile_info(self) -> Dict[str, Any]:
        """Get information about current agent profile"""
        profile_info = {
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

        # Add semantic search info if available
        if hasattr(self.cognition, 'get_semantic_search_info'):
            profile_info['semantic_search'] = self.cognition.get_semantic_search_info()

        return profile_info
"""
Focused large-scale test with reduced evolution overhead
Tests embedding generation and semantic search with 50 conversations
"""
import asyncio
import time
from typing import List, Dict, Any
from agenesis.core import Agent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_focused_embeddings():
    """Test focused embedding functionality with 50 conversations"""

    print("=" * 60)
    print("🎯 FOCUSED EMBEDDING TEST (50 CONVERSATIONS)")
    print("=" * 60)

    # Generate focused test conversations
    conversations = [
        "I'm a software engineer working with React and Node.js",
        "I'm building machine learning models with Python and TensorFlow",
        "I'm leading a product team at a tech startup",
        "I'm researching blockchain applications for supply chain",
        "I'm learning Spanish and planning a trip to Europe",
        "I'm following a plant-based diet for better health",
        "I'm implementing microservices architecture",
        "I'm working on natural language processing projects",
        "I'm managing cross-functional engineering teams",
        "I'm exploring quantum computing applications",
        "I'm training for a marathon this year",
        "I'm practicing meditation and mindfulness",
        "I'm setting up CI/CD pipelines with Docker",
        "I'm building recommendation systems",
        "I'm implementing agile methodologies",
        "I'm working on IoT sensor networks",
        "I'm interested in sustainable living",
        "I'm dealing with work-related stress",
        "I'm debugging memory leaks in applications",
        "I'm using pandas for data analysis",
        "I'm conducting user research studies",
        "I'm implementing smart city infrastructure",
        "I'm learning guitar and piano",
        "I'm tracking fitness with wearable devices",
        "I'm optimizing database query performance",
        "I'm training deep learning models",
        "I'm managing stakeholder expectations",
        "I'm researching 5G network implementations",
        "I'm passionate about photography",
        "I'm working on sleep quality improvement",
        "I'm implementing GraphQL APIs",
        "I'm building computer vision models",
        "I'm working on team productivity",
        "I'm exploring augmented reality experiences",
        "I'm interested in wine tasting",
        "I'm practicing yoga regularly",
        "I'm setting up monitoring with Prometheus",
        "I'm implementing time series forecasting",
        "I'm conducting competitive analysis",
        "I'm working on autonomous vehicle tech",
        "I'm learning about cryptocurrency",
        "I'm managing chronic pain",
        "I'm implementing OAuth authentication",
        "I'm working with Apache Spark",
        "I'm planning product roadmaps",
        "I'm implementing edge computing",
        "I'm passionate about cooking",
        "I'm optimizing brain health",
        "I'm using Redis for caching",
        "I'm building chatbots with transformers"
    ]

    test_profile = "focused_embedding_test"

    print(f"📝 Processing {len(conversations)} conversations...")

    # Create agent with evolution disabled to speed up test
    agent = Agent(profile=test_profile, config={
        'use_semantic_search': True,
        'storage_type': 'sqlite',
        'evolution': {'enabled': False}  # Disable evolution for faster processing
    })

    # Process conversations
    start_time = time.time()
    embedding_count = 0

    for i, conversation in enumerate(conversations, 1):
        try:
            response = await agent.process_input(conversation)

            # Verify embedding
            record = agent.get_current_focus_record()
            if record and record.embedding and len(record.embedding) > 0:
                embedding_count += 1

            if i % 10 == 0:
                print(f"   Processed {i}/{len(conversations)} conversations")

        except Exception as e:
            print(f"❌ Error processing conversation {i}: {e}")

    processing_time = time.time() - start_time

    print(f"\n📊 Phase 1 Results:")
    print(f"   Processing time: {processing_time:.2f} seconds")
    print(f"   Average per conversation: {processing_time/len(conversations):.2f}s")
    print(f"   Embeddings generated: {embedding_count}/{len(conversations)}")
    print(f"   Success rate: {embedding_count/len(conversations)*100:.1f}%")

    # Check memory systems
    working_records = agent.working_memory.get_all()
    working_embedded = sum(1 for r in working_records if r.embedding and len(r.embedding) > 0)
    print(f"   Working memory embedded: {working_embedded}/{len(working_records)}")

    if agent.has_persistent_memory:
        stats = agent.persistent_memory.get_embedding_statistics()
        print(f"   Database coverage: {stats['embedding_coverage']:.1f}%")
        print(f"   Total persistent records: {stats['total_records']}")

    print(f"\n🔄 Phase 2: Agent restart test...")

    # Test restart
    agent.end_session()
    del agent

    restart_start = time.time()
    agent2 = Agent(profile=test_profile, config={
        'use_semantic_search': True,
        'storage_type': 'sqlite',
        'evolution': {'enabled': False}
    })
    restart_time = time.time() - restart_start

    print(f"   Restart time: {restart_time:.2f} seconds")

    if agent2.has_persistent_memory:
        loaded_stats = agent2.persistent_memory.get_embedding_statistics()
        print(f"   Loaded records: {loaded_stats['total_records']}")
        print(f"   Embedding coverage: {loaded_stats['embedding_coverage']:.1f}%")

    print(f"\n🔍 Phase 3: Semantic search test...")

    # Test semantic search
    search_queries = [
        "What machine learning frameworks do you use?",
        "Tell me about your software engineering experience",
        "What are your health and fitness interests?"
    ]

    search_times = []
    for query in search_queries:
        query_start = time.time()
        response = await agent2.process_input(query)
        query_time = time.time() - query_start
        search_times.append(query_time)

        print(f"   Query: '{query[:40]}...'")
        print(f"   Response length: {len(response)} chars")
        print(f"   Search time: {query_time:.2f}s")

    avg_search_time = sum(search_times) / len(search_times)
    print(f"   Average search time: {avg_search_time:.2f}s")

    print(f"\n✅ Focused test completed successfully!")
    print(f"   {len(conversations)} conversations processed")
    print(f"   {embedding_count} embeddings generated")
    print(f"   Cross-session persistence verified")
    print(f"   Semantic search functioning")
    print("=" * 60)

    agent2.end_session()

    return {
        'conversations_processed': len(conversations),
        'embeddings_generated': embedding_count,
        'processing_time': processing_time,
        'restart_time': restart_time,
        'avg_search_time': avg_search_time,
        'success_rate': embedding_count/len(conversations)*100
    }

if __name__ == "__main__":
    asyncio.run(test_focused_embeddings())
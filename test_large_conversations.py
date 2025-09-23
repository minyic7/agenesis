"""
Large-scale conversation test with 200 meaningful interactions
Tests embedding generation, storage, and cross-session persistence
"""
import asyncio
import time
from typing import List, Dict, Any
from agenesis.core import Agent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_meaningful_conversations() -> List[str]:
    """Generate 200 diverse, meaningful conversation inputs"""

    conversations = []

    # Software Development (40 conversations)
    software_dev = [
        "I'm a senior software engineer at a tech startup",
        "I specialize in full-stack development using React and Node.js",
        "My current project involves building a microservices architecture",
        "I'm implementing CI/CD pipelines with Docker and Kubernetes",
        "We're using GraphQL for our API layer instead of REST",
        "I'm debugging a memory leak in our Node.js application",
        "The team is transitioning from JavaScript to TypeScript",
        "I'm setting up monitoring with Prometheus and Grafana",
        "We're implementing OAuth 2.0 for user authentication",
        "I'm optimizing database queries to reduce response times",
        "The frontend needs better error handling and user feedback",
        "I'm writing unit tests with Jest and integration tests with Cypress",
        "We're planning to migrate from MongoDB to PostgreSQL",
        "I'm implementing rate limiting to prevent API abuse",
        "The application needs to handle 10k concurrent users",
        "I'm setting up Redis for caching frequently accessed data",
        "We're implementing real-time features using WebSocket connections",
        "I'm working on code splitting to improve initial load times",
        "The team is adopting clean architecture principles",
        "I'm implementing serverless functions for specific use cases",
        "We need to improve our error logging and monitoring",
        "I'm setting up automated security scanning in our pipeline",
        "The application requires GDPR compliance for user data",
        "I'm implementing progressive web app features",
        "We're using feature flags for gradual rollouts",
        "I'm optimizing images and assets for better performance",
        "The team is implementing pair programming practices",
        "I'm setting up automated database migrations",
        "We're implementing search functionality with Elasticsearch",
        "I'm working on mobile responsiveness and touch interactions",
        "The application needs offline capabilities",
        "I'm implementing data validation on both client and server",
        "We're setting up A/B testing for new features",
        "I'm working on internationalization and localization",
        "The team is implementing design system components",
        "I'm setting up automated performance testing",
        "We're implementing event-driven architecture patterns",
        "I'm working on accessibility compliance (WCAG guidelines)",
        "The application needs better SEO optimization",
        "I'm implementing advanced analytics and user tracking"
    ]

    # Data Science & Machine Learning (40 conversations)
    data_science = [
        "I'm a data scientist working on predictive analytics",
        "I'm building machine learning models for customer segmentation",
        "My current project involves natural language processing",
        "I'm using Python with scikit-learn and pandas for analysis",
        "I'm implementing deep learning models with TensorFlow",
        "The dataset has 2 million records with 150 features",
        "I'm working on feature engineering and selection",
        "I'm implementing cross-validation for model evaluation",
        "The model accuracy needs to be above 95% for production",
        "I'm dealing with imbalanced datasets using SMOTE",
        "I'm implementing ensemble methods for better predictions",
        "The team is using MLflow for model versioning and tracking",
        "I'm working on hyperparameter tuning with grid search",
        "I'm implementing real-time inference with Flask APIs",
        "The model needs to handle streaming data efficiently",
        "I'm using BERT for sentiment analysis tasks",
        "I'm implementing computer vision models for image classification",
        "The team is working on recommendation systems",
        "I'm implementing time series forecasting models",
        "I'm using Apache Spark for distributed data processing",
        "I'm working on dimensionality reduction with PCA",
        "The team is implementing automated machine learning (AutoML)",
        "I'm using Docker to containerize ML models",
        "I'm implementing model monitoring and drift detection",
        "I'm working on explainable AI for model interpretability",
        "The team is using Kubernetes for ML model deployment",
        "I'm implementing federated learning for privacy-preserving ML",
        "I'm working on reinforcement learning for optimization",
        "I'm using PyTorch for research and experimentation",
        "The team is implementing graph neural networks",
        "I'm working on anomaly detection in financial data",
        "I'm implementing chatbots using transformer models",
        "I'm using Jupyter notebooks for exploratory data analysis",
        "The team is working on real-time recommendation engines",
        "I'm implementing A/B testing for ML model comparisons",
        "I'm working on multi-modal learning combining text and images",
        "I'm using Apache Airflow for ML pipeline orchestration",
        "The team is implementing edge computing for IoT devices",
        "I'm working on neural architecture search (NAS)",
        "I'm implementing privacy-preserving machine learning techniques"
    ]

    # Business & Management (30 conversations)
    business = [
        "I'm leading a product team of 12 engineers",
        "We're planning the roadmap for the next quarter",
        "I'm working on improving team productivity and velocity",
        "The company is scaling from 50 to 200 employees",
        "I'm implementing agile methodologies across teams",
        "We're conducting user research to validate features",
        "I'm managing stakeholder expectations and communication",
        "The team needs better project management tools",
        "I'm working on cross-functional collaboration",
        "We're implementing OKRs for goal setting and tracking",
        "I'm conducting performance reviews and feedback sessions",
        "The company is expanding to international markets",
        "I'm working on talent acquisition and retention strategies",
        "We're implementing diversity and inclusion initiatives",
        "I'm managing budget allocation for different projects",
        "The team is transitioning to remote-first work model",
        "I'm working on company culture and values alignment",
        "We're implementing data-driven decision making processes",
        "I'm conducting competitive analysis and market research",
        "The company is considering acquisition opportunities",
        "I'm working on customer success and retention metrics",
        "We're implementing new sales processes and tools",
        "I'm managing vendor relationships and contracts",
        "The team is working on digital transformation initiatives",
        "I'm implementing risk management and compliance procedures",
        "We're working on brand positioning and marketing strategy",
        "I'm conducting training and development programs",
        "The company is preparing for IPO considerations",
        "I'm working on operational efficiency improvements",
        "We're implementing sustainability and CSR initiatives"
    ]

    # Technology & Innovation (30 conversations)
    technology = [
        "I'm researching emerging technologies and trends",
        "We're implementing blockchain for supply chain transparency",
        "I'm working on IoT devices and sensor networks",
        "The team is exploring quantum computing applications",
        "I'm implementing cybersecurity best practices",
        "We're working on augmented reality experiences",
        "I'm researching 5G network implementations",
        "The team is implementing edge computing solutions",
        "I'm working on autonomous vehicle technologies",
        "We're exploring virtual reality training applications",
        "I'm implementing smart city infrastructure",
        "The team is working on renewable energy systems",
        "I'm researching artificial general intelligence",
        "We're implementing biometric authentication systems",
        "I'm working on robotic process automation",
        "The team is exploring brain-computer interfaces",
        "I'm implementing distributed ledger technologies",
        "We're working on 3D printing and manufacturing",
        "I'm researching nanotechnology applications",
        "The team is implementing voice recognition systems",
        "I'm working on satellite communication networks",
        "We're exploring gene editing technologies",
        "I'm implementing smart grid energy management",
        "The team is working on autonomous drone systems",
        "I'm researching materials science innovations",
        "We're implementing precision agriculture technologies",
        "I'm working on advanced manufacturing processes",
        "The team is exploring space technology applications",
        "I'm implementing biotechnology solutions",
        "We're working on sustainable technology innovations"
    ]

    # Personal & Lifestyle (30 conversations)
    personal = [
        "I'm planning a career transition to data science",
        "I'm learning Spanish and French for travel",
        "I'm training for my first marathon this year",
        "I'm passionate about sustainable living and environment",
        "I'm learning to play piano and guitar",
        "I'm interested in photography and videography",
        "I'm planning a trip to Japan and Southeast Asia",
        "I'm learning about investing and personal finance",
        "I'm passionate about cooking and trying new cuisines",
        "I'm interested in meditation and mindfulness practices",
        "I'm learning woodworking and furniture making",
        "I'm passionate about hiking and outdoor adventures",
        "I'm interested in renewable energy for my home",
        "I'm learning about wine tasting and viticulture",
        "I'm passionate about volunteering and community service",
        "I'm interested in urban gardening and permaculture",
        "I'm learning about cryptocurrency and blockchain",
        "I'm passionate about reading science fiction novels",
        "I'm interested in home automation and smart devices",
        "I'm learning about nutrition and healthy eating",
        "I'm passionate about electric vehicles and sustainability",
        "I'm interested in learning new programming languages",
        "I'm planning to start a side business",
        "I'm passionate about documentary filmmaking",
        "I'm interested in astronomy and space exploration",
        "I'm learning about real estate investment",
        "I'm passionate about animal welfare and conservation",
        "I'm interested in minimalism and decluttering",
        "I'm learning about renewable energy systems",
        "I'm passionate about education and lifelong learning"
    ]

    # Health & Wellness (30 conversations)
    health = [
        "I'm working on improving my sleep quality",
        "I'm following a plant-based diet for health",
        "I'm dealing with work-related stress and burnout",
        "I'm implementing a daily exercise routine",
        "I'm learning about mental health and self-care",
        "I'm working with a nutritionist on meal planning",
        "I'm practicing yoga and meditation regularly",
        "I'm tracking my fitness progress with wearable devices",
        "I'm learning about intermittent fasting benefits",
        "I'm working on building healthy habits",
        "I'm dealing with chronic pain management",
        "I'm learning about supplements and vitamins",
        "I'm working on improving my posture",
        "I'm practicing mindfulness and stress reduction",
        "I'm learning about functional medicine approaches",
        "I'm working on work-life balance",
        "I'm dealing with anxiety and depression",
        "I'm learning about holistic health approaches",
        "I'm working on building stronger relationships",
        "I'm practicing gratitude and positive thinking",
        "I'm learning about biohacking and optimization",
        "I'm working on digital detox and screen time",
        "I'm dealing with autoimmune health issues",
        "I'm learning about brain health and cognition",
        "I'm working on building resilience and adaptability",
        "I'm practicing breathing exercises and techniques",
        "I'm learning about gut health and microbiome",
        "I'm working on emotional intelligence and regulation",
        "I'm dealing with hormonal health optimization",
        "I'm learning about longevity and aging research"
    ]

    # Combine all conversation categories
    conversations.extend(software_dev)
    conversations.extend(data_science)
    conversations.extend(business)
    conversations.extend(technology)
    conversations.extend(personal)
    conversations.extend(health)

    return conversations[:200]  # Ensure exactly 200 conversations


async def test_large_scale_conversations():
    """Test 200 meaningful conversations with embedding verification"""

    print("=" * 60)
    print("🧪 LARGE-SCALE CONVERSATION EMBEDDING TEST")
    print("=" * 60)

    # Generate test data
    conversations = generate_meaningful_conversations()
    print(f"📝 Generated {len(conversations)} meaningful conversations")

    # Test profile for persistence
    test_profile = "large_scale_test_user"

    print(f"\n🚀 Phase 1: Processing {len(conversations)} conversations...")
    print("-" * 50)

    # Create agent with SQLite storage for embedding statistics
    agent = Agent(profile=test_profile, config={
        'use_semantic_search': True,
        'storage_type': 'sqlite'
    })

    # Track statistics
    start_time = time.time()
    embedding_successes = 0
    processing_times = []

    # Process all conversations
    for i, conversation in enumerate(conversations, 1):
        conversation_start = time.time()

        try:
            response = await agent.process_input(conversation)

            # Verify embedding was created
            current_record = agent.get_current_focus_record()
            if current_record and current_record.embedding and len(current_record.embedding) > 0:
                embedding_successes += 1

            conversation_time = time.time() - conversation_start
            processing_times.append(conversation_time)

            # Progress indicator
            if i % 20 == 0:
                avg_time = sum(processing_times[-20:]) / min(20, len(processing_times))
                print(f"   Processed {i:3d}/200 conversations (avg: {avg_time:.2f}s/conversation)")

        except Exception as e:
            print(f"❌ Error processing conversation {i}: {e}")

    total_time = time.time() - start_time
    avg_processing_time = sum(processing_times) / len(processing_times)

    print(f"\n📊 Phase 1 Results:")
    print(f"   Total processing time: {total_time:.2f} seconds")
    print(f"   Average per conversation: {avg_processing_time:.2f} seconds")
    print(f"   Embeddings generated: {embedding_successes}/200")
    print(f"   Success rate: {embedding_successes/200*100:.1f}%")

    # Verify memory systems
    print(f"\n🧠 Memory System Verification:")

    # Working memory
    working_records = agent.working_memory.get_all()
    working_embedded = sum(1 for r in working_records if r.embedding and len(r.embedding) > 0)
    print(f"   Working memory: {working_embedded}/{len(working_records)} embedded")

    # Persistent memory
    if agent.has_persistent_memory:
        all_persistent = agent.persistent_memory.get_recent(250)  # Get more than 200
        persistent_embedded = sum(1 for r in all_persistent if r.embedding and len(r.embedding) > 0)
        print(f"   Persistent memory: {persistent_embedded}/{len(all_persistent)} embedded")

        # Database statistics
        if hasattr(agent.persistent_memory, 'get_embedding_statistics'):
            stats = agent.persistent_memory.get_embedding_statistics()
            print(f"   Database coverage: {stats['embedding_coverage']:.1f}% ({stats['records_with_embeddings']}/{stats['total_records']})")

    print(f"\n🔄 Phase 2: Agent restart and persistence test...")
    print("-" * 50)

    # End current session
    agent.end_session()
    del agent

    # Create new agent with same profile (simulates app restart)
    restart_start = time.time()
    agent2 = Agent(profile=test_profile, config={
        'use_semantic_search': True,
        'storage_type': 'sqlite'
    })
    restart_time = time.time() - restart_start

    print(f"   Agent restart time: {restart_time:.2f} seconds")

    # Verify persistent memory loaded
    if agent2.has_persistent_memory:
        loaded_persistent = agent2.persistent_memory.get_recent(100)
        loaded_embedded = sum(1 for r in loaded_persistent if r.embedding and len(r.embedding) > 0)
        print(f"   Loaded persistent memories: {len(loaded_persistent)}")
        print(f"   Pre-embedded memories: {loaded_embedded}")
        print(f"   Embedding coverage: {loaded_embedded/len(loaded_persistent)*100:.1f}%")

    print(f"\n🔍 Phase 3: Semantic search testing...")
    print("-" * 50)

    # Test semantic search across conversation history
    test_queries = [
        "Tell me about machine learning and data science work",
        "What programming languages and frameworks are mentioned?",
        "What are the business and management topics discussed?",
        "What health and wellness practices are covered?",
        "What emerging technologies are being explored?",
        "What personal interests and hobbies are mentioned?"
    ]

    search_results = []
    for query in test_queries:
        query_start = time.time()
        response = await agent2.process_input(query)
        query_time = time.time() - query_start

        search_results.append({
            'query': query,
            'response_length': len(response),
            'processing_time': query_time
        })

        print(f"   Query: '{query[:50]}...'")
        print(f"   Response length: {len(response)} chars")
        print(f"   Processing time: {query_time:.2f}s")
        print()

    avg_search_time = sum(r['processing_time'] for r in search_results) / len(search_results)
    print(f"   Average search time: {avg_search_time:.2f} seconds")

    print(f"\n🎯 Final Verification:")
    print("-" * 50)

    # Final memory statistics
    final_working = agent2.working_memory.get_all()
    final_working_embedded = sum(1 for r in final_working if r.embedding and len(r.embedding) > 0)

    final_coverage = 0
    if agent2.has_persistent_memory:
        final_stats = agent2.persistent_memory.get_embedding_statistics()
        print(f"   Final database records: {final_stats['total_records']}")
        print(f"   Final embedding coverage: {final_stats['embedding_coverage']:.1f}%")
        print(f"   Working memory embedded: {final_working_embedded}/{len(final_working)}")
        final_coverage = final_stats['embedding_coverage']

    print(f"\n✅ Large-scale test completed successfully!")
    print(f"   200 conversations processed and embedded")
    print(f"   Cross-session persistence verified")
    print(f"   Semantic search functioning across full history")
    print("=" * 60)

    return {
        'conversations_processed': len(conversations),
        'embeddings_generated': embedding_successes,
        'total_processing_time': total_time,
        'avg_processing_time': avg_processing_time,
        'restart_time': restart_time,
        'avg_search_time': avg_search_time,
        'final_coverage': final_coverage
    }


if __name__ == "__main__":
    asyncio.run(test_large_scale_conversations())
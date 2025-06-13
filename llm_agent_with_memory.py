"""
Enhanced LLM Agent with Memory and Persistent Context
Makes agents more human-like by giving them:
1. Personal identity and characteristics
2. Memory of past decisions and experiences
3. Relationships with neighbors
4. Cumulative neighborhood history
"""

import random
import json
from datetime import datetime
from Agent import Agent
import config as cfg

class LLMAgentWithMemory(Agent):
    """LLM Agent that maintains memory and personal context"""
    
    def __init__(self, type_id, scenario='baseline', llm_model=None, llm_url=None, llm_api_key=None):
        super().__init__(type_id)
        self.scenario = scenario
        self.llm_model = llm_model or cfg.OLLAMA_MODEL
        self.llm_url = llm_url or cfg.OLLAMA_URL
        self.llm_api_key = llm_api_key or cfg.OLLAMA_API_KEY
        
        # Initialize personal identity
        self.personal_id = random.randint(1000, 9999)
        self.initialize_personal_context(type_id, scenario)
        
        # Initialize memory systems
        self.move_history = []  # List of past moves with reasons
        self.neighborhood_experiences = []  # Memorable events
        self.neighbor_relationships = {}  # Track specific neighbors
        self.satisfaction_history = []  # Track satisfaction over time
        self.time_in_current_location = 0
        self.total_moves = 0
        
    def initialize_personal_context(self, type_id, scenario):
        """Create a persistent personal identity for this agent"""
        
        # Base identity from scenario
        if scenario == 'baseline':
            self.identity = {
                'type': 'red team' if type_id == 0 else 'blue team',
                'tolerance_level': random.choice(['low', 'moderate', 'high']),
                'stability_preference': random.choice(['likes_change', 'neutral', 'prefers_stability']),
                'social_preference': random.choice(['very_social', 'moderate', 'private'])
            }
        
        elif scenario == 'race_white_black':
            if type_id == 0:
                children_count = random.choice([0, 1, 2, 3])
                self.identity = {
                    'type': 'white middle class family',
                    'children': children_count,
                    'children_ages': self._generate_children_ages(children_count),
                    'homeowner_status': random.choice(['owner', 'renter']),
                    'years_in_area': random.randint(0, 20),
                    'extended_family_nearby': random.choice([True, False]),
                    'primary_concerns': random.sample(['schools', 'property_values', 'safety', 'community'], 2)
                }
            else:
                children_count = random.choice([0, 1, 2, 3])
                self.identity = {
                    'type': 'Black family',
                    'children': children_count,
                    'children_ages': self._generate_children_ages(children_count),
                    'homeowner_status': random.choice(['owner', 'renter', 'seeking_to_buy']),
                    'church_affiliation': random.choice([True, False]),
                    'extended_family_nearby': random.choice([True, False]),
                    'primary_concerns': random.sample(['community', 'cultural_connection', 'schools', 'affordability'], 2)
                }
                
        elif scenario == 'economic_high_working':
            if type_id == 0:
                self.identity = {
                    'type': 'high-income household',
                    'profession': random.choice(['tech_executive', 'doctor', 'lawyer', 'business_owner']),
                    'investment_mindset': random.choice(['aggressive', 'moderate', 'conservative']),
                    'lifestyle_priorities': random.sample(['exclusivity', 'amenities', 'networking', 'privacy'], 2),
                    'property_portfolio': random.randint(1, 4)
                }
            else:
                self.identity = {
                    'type': 'working-class household',
                    'job_stability': random.choice(['stable', 'gig_economy', 'multiple_jobs']),
                    'commute_sensitivity': random.choice(['very_high', 'high', 'moderate']),
                    'support_network_importance': random.choice(['critical', 'very_important', 'important']),
                    'financial_stress': random.choice(['high', 'moderate', 'manageable'])
                }
        
        else:
            # Default fallback
            self.identity = {
                'type': 'resident',
                'tolerance_level': random.choice(['low', 'moderate', 'high']),
                'stability_preference': random.choice(['likes_change', 'neutral', 'prefers_stability'])
            }
                
        # Add personality traits common to all
        self.identity.update({
            'personality': random.choice(['optimistic', 'pragmatic', 'cautious']),
            'decision_style': random.choice(['quick', 'deliberate', 'consultative']),
            'neighbor_memory_strength': random.choice(['excellent', 'good', 'average'])
        })
        
    def _generate_children_ages(self, children_count=None):
        """Generate realistic children ages if applicable"""
        if children_count is None or children_count == 0:
            return []
        return sorted([random.randint(0, 18) for _ in range(children_count)])
    
    def remember_move(self, from_pos, to_pos, reason, neighborhood_snapshot):
        """Record a move in agent's memory"""
        self.move_history.append({
            'step': len(self.move_history),
            'from': from_pos,
            'to': to_pos,
            'reason': reason,
            'neighborhood_before': neighborhood_snapshot,
            'timestamp': datetime.now().isoformat()
        })
        self.total_moves += 1
        self.time_in_current_location = 0
        
    def remember_staying(self, current_pos, reason, satisfaction_level):
        """Record decision to stay in current location"""
        self.satisfaction_history.append({
            'step': len(self.satisfaction_history),
            'position': current_pos,
            'satisfaction': satisfaction_level,
            'reason': reason
        })
        self.time_in_current_location += 1
        
    def remember_neighbor_interaction(self, neighbor_type, interaction_quality):
        """Build relationships with specific neighbors over time"""
        if neighbor_type not in self.neighbor_relationships:
            self.neighbor_relationships[neighbor_type] = {
                'first_encounter': len(self.move_history) + len(self.satisfaction_history),
                'interactions': []
            }
        
        self.neighbor_relationships[neighbor_type]['interactions'].append({
            'quality': interaction_quality,  # positive, neutral, negative
            'step': len(self.move_history) + len(self.satisfaction_history)
        })
        
    def add_neighborhood_experience(self, experience):
        """Record a memorable neighborhood event"""
        self.neighborhood_experiences.append({
            'event': experience,
            'step': len(self.move_history) + len(self.satisfaction_history),
            'impact': random.choice(['positive', 'neutral', 'negative'])
        })
        
    def get_memory_summary(self, max_memories=5):
        """Summarize agent's recent memories for LLM context"""
        summary = []
        
        # Recent moves
        if self.move_history:
            recent_moves = self.move_history[-2:]
            for move in recent_moves:
                summary.append(f"Previously moved because: {move['reason']}")
                
        # Time in current location
        if self.time_in_current_location > 0:
            summary.append(f"Have been in current location for {self.time_in_current_location} periods")
            
        # Recent satisfaction
        if self.satisfaction_history:
            recent_satisfaction = self.satisfaction_history[-3:]
            avg_satisfaction = sum(s['satisfaction'] for s in recent_satisfaction) / len(recent_satisfaction)
            summary.append(f"Recent satisfaction level: {avg_satisfaction:.1f}/10")
            
        # Neighbor relationships
        if self.neighbor_relationships:
            for neighbor_type, relationship in self.neighbor_relationships.items():
                if relationship['interactions']:
                    recent = relationship['interactions'][-3:]
                    positive = sum(1 for i in recent if i['quality'] == 'positive')
                    summary.append(f"Recent interactions with {neighbor_type}: {positive}/3 positive")
                    
        # Notable experiences
        if self.neighborhood_experiences:
            recent_exp = self.neighborhood_experiences[-2:]
            for exp in recent_exp:
                summary.append(f"Recent event: {exp['event']} ({exp['impact']})")
                
        return summary[:max_memories]
    
    def get_identity_context(self):
        """Format agent's identity for LLM prompt"""
        identity_parts = []
        
        # Core identity
        identity_parts.append(f"You are a {self.identity['type']}")
        
        # Relevant personal details
        if 'children' in self.identity and self.identity['children'] > 0:
            identity_parts.append(f"You have {self.identity['children']} children (ages: {', '.join(map(str, self.identity['children_ages']))})")
            
        if 'profession' in self.identity:
            identity_parts.append(f"You work as a {self.identity['profession']}")
            
        if 'primary_concerns' in self.identity:
            identity_parts.append(f"Your main priorities are: {', '.join(self.identity['primary_concerns'])}")
            
        # Personality
        identity_parts.append(f"You tend to be {self.identity['personality']} and make decisions in a {self.identity['decision_style']} manner")
        
        # Stability preference
        if 'stability_preference' in self.identity:
            if self.identity['stability_preference'] == 'prefers_stability':
                identity_parts.append("You prefer stability and don't like moving often")
            elif self.identity['stability_preference'] == 'likes_change':
                identity_parts.append("You're comfortable with change and don't mind moving")
                
        return ". ".join(identity_parts)
    
    def analyze_neighborhood_change(self, old_context, new_context):
        """Detect and remember significant neighborhood changes"""
        # Compare old and new neighborhood to detect changes
        changes = []
        
        for i in range(3):
            for j in range(3):
                if old_context[i][j] != new_context[i][j]:
                    if old_context[i][j] == 'S' and new_context[i][j] == 'O':
                        changes.append("A neighbor like you moved away")
                    elif old_context[i][j] == 'O' and new_context[i][j] == 'S':
                        changes.append("A new neighbor like you moved in")
                    elif old_context[i][j] == 'E' and new_context[i][j] != 'E':
                        changes.append("Someone moved into an empty house nearby")
                        
        return changes
    
    def get_enhanced_prompt(self, context_str, r, c, grid):
        """Create an enhanced prompt including memory and identity"""
        
        # Get memory summary
        memories = self.get_memory_summary()
        memory_context = "\n".join(f"- {memory}" for memory in memories) if memories else "- No significant past experiences yet"
        
        # Get identity context  
        identity_context = self.get_identity_context()
        
        # Build enhanced prompt
        enhanced_prompt = f"""{identity_context}

Your current 3x3 neighborhood:
{context_str}

Your memories and experiences:
{memory_context}

You've lived here for {self.time_in_current_location} time periods.
You've moved {self.total_moves} times total.

Considering your personal history, family situation, and accumulated experiences in this neighborhood, 
would you like to move to a different house or stay where you are?

Think about:
- Your past experiences and satisfaction
- Your relationships with neighbors
- Your family's specific needs
- How long you've been in your current location
- Whether moving would actually improve your situation

Respond with ONLY:
- The coordinates (row, col) of an empty house you'd move to, OR
- None (if you prefer to stay)

Your decision:"""
        
        return enhanced_prompt
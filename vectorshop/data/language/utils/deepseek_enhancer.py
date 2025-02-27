"""
Deepseek enhancer
DeepSeek-R1 query enhancement and reranking module for the VectorShop project.
"""

import torch
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM 

class DeepSeekEnhancer:
    """
    Enhances search relevance using DeepSeek-R1-Distill-Qwen model
    for query understanding and result reranking.
    """
    
    def __init__(self, device="cpu"):
        """Initialize the DeepSeek enhancer with the specified device."""
        self.device = device
        self._model = None
        self._tokenizer = None
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        
    def load_model(self):
        """Lazily load the DeepSeek model and tokenizer."""
        if self._model is None:
            print(f"Loading {self.model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            ).to(self.device)
            self._model.eval()  # Set to evaluation mode
            print("DeepSeek model loaded successfully")
        return self._model, self._tokenizer
    

    def optimize_memory(self):
        """
        Apply memory optimizations to reduce GPU memory usage.
        Call this after loading the model.
        """
        import gc
        import torch
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # If model is already loaded, apply memory optimizations
        if self._model is not None:
            # Use CPU offloading for parts of the model when possible
            if self.device == "cuda":
                # Move less frequently used components to CPU
                for param in self._model.parameters():
                    if not param.requires_grad:
                        param.data = param.data.to("cpu")
                        if param._grad is not None:
                            param._grad.data = param._grad.data.to("cpu")
        
        return True

    def free_memory(self):
        """
        Free up memory by moving model to CPU.
        Call this after completing intensive operations.
        """
        import gc
        import torch
        
        # Move model to CPU if it's on GPU
        if self._model is not None and next(self._model.parameters()).device.type == "cuda":
            self._model = self._model.to("cpu")
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def rerank_results_with_memory_management(self, 
                                            query: str, 
                                            results: pd.DataFrame, 
                                            query_analysis: Dict = None,
                                            top_n: int = 10) -> pd.DataFrame:
        """
        Memory-efficient version of rerank_results.
        Processes in smaller batches and cleans up memory.
        
        Args:
            query: Original search query
            results: DataFrame containing search results
            query_analysis: Optional pre-analyzed query information
            top_n: Maximum number of results to rerank
            
        Returns:
            DataFrame with reranked results
        """
        if len(results) == 0:
            return results
            
        # Limit to top_n results for efficiency
        results_to_rerank = results.head(top_n).copy()
        
        # Get query analysis if not provided
        if query_analysis is None:
            query_analysis = self.analyze_query(query)
        
        model, tokenizer = self.load_model()
        
        # Apply memory optimizations
        self.optimize_memory()
        
        # Smaller batch size for memory efficiency
        batch_size = 5
        
        # Process in batches
        all_semantic_scores = []
        
        for batch_start in range(0, len(results_to_rerank), batch_size):
            batch_end = min(batch_start + batch_size, len(results_to_rerank))
            batch = results_to_rerank.iloc[batch_start:batch_end]
            
            batch_scores = []
            
            for _, row in batch.iterrows():
                # Format product information (same as original)
                product_desc = f"""
                Name: {row['product_name']}
                Category: {row['category']}
                Price: {row['price_usd']} USD
                """
                
                # Add product description
                if 'about_product' in row and not pd.isna(row['about_product']):
                    product_desc += f"Description: {row['about_product'][:300]}...\n"
                
                # Create prompt with query context
                prompt = f"""
                Rate how relevant this product is to the search query on a scale from 0 to 10.
                
                Search Query: "{query}"
                
                Product Information:
                {product_desc}
                
                Key search requirements:
                - Product Type: {query_analysis.get('product_type', 'Any')}
                - Important Features: {', '.join(query_analysis.get('key_features', []))}
                - Price Constraint: {query_analysis.get('price_constraint', 'None')}
                
                Consider exact category matches, feature matches, and price constraints.
                Output only the numerical score (0-10).
                """
                
                # Generate rating
                try:
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=10,
                            temperature=0.2,
                            do_sample=True
                        )
                    
                    # Extract numerical rating
                    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                    
                    # Parse the rating
                    parsed_result = self._parse_deepseek_response(raw_output, default_value={"score": 0})
                    if isinstance(parsed_result, dict) and "score" in parsed_result:
                        score = parsed_result["score"]
                    else:
                        import re
                        match = re.search(r'(\d+(\.\d+)?)', raw_output)
                        score = float(match.group(1)) if match else 0.0
                    
                    score = min(max(score, 0.0), 10.0)  # Ensure in range 0-10
                except Exception as e:
                    print(f"Error generating score: {e}")
                    score = 0.0
                    
                batch_scores.append(score)
                
                # Cleanup after each item to prevent memory accumulation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            all_semantic_scores.extend(batch_scores)
            
            # Force memory cleanup after each batch
            self.free_memory()
        
        # Add semantic scores to results
        results_to_rerank['semantic_score'] = all_semantic_scores
        
        # Rest of the function remains the same as original rerank_results
        # Create a deep copy to ensure values don't get overwritten
        reranked_results = results_to_rerank.copy(deep=True)
        
        # Normalize original scores to 0-10 range
        if 'hybrid_score' in reranked_results.columns and len(reranked_results) > 1:
            min_score = reranked_results['hybrid_score'].min()
            max_score = reranked_results['hybrid_score'].max()
            if max_score > min_score:
                reranked_results['normalized_score'] = (
                    (reranked_results['hybrid_score'] - min_score) / 
                    (max_score - min_score) * 10
                )
            else:
                reranked_results['normalized_score'] = reranked_results['hybrid_score']
        else:
            # Fallback if hybrid_score isn't available
            reranked_results['normalized_score'] = reranked_results.get('hybrid_score', 0)
        
        # Calculate final score (weighted combination)
        reranked_results['final_score'] = (
            reranked_results['normalized_score'].fillna(0) * 0.3 + 
            reranked_results['semantic_score'].fillna(0) * 0.7
        )
        
        # Sort by final score
        reranked_results = reranked_results.sort_values('final_score', ascending=False)
        
        # Clean up one last time
        self.free_memory()
        
        return reranked_results
    
    def analyze_query(self, query: str) -> Dict:
        """
        Extract structured information from the search query using DeepSeek.
        
        Args:
            query: The search query text
            
        Returns:
            Dictionary with product type, key features, and price constraints
        """
        model, tokenizer = self.load_model()
        
        # Prepare the prompt for query analysis
        prompt = f"""
        Analyze this e-commerce search query to extract structured information.
        
        Query: "{query}"
        
        Provide only a JSON object with these fields:
        - product_type: The main product category (like cable, headphone, tv)
        - key_features: List of important features/attributes requested (be sure to include 'warranty' if mentioned)
        - price_constraint: Any price constraints
        
        Format as valid JSON only - no additional text.
        """
        
        # Generate analysis
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # special handling for iPhone cables
        query_lower = query.lower()
        if "iphone" in query_lower and any(word in query_lower for word in ["cable", "charger", "charging"]):
            # Special handling for iPhone cables
            special_boost = {
                "B08CF3B7N1": 3.0,  # Portronics cable
            }
            # Get the query info as usual
            query_info = self._extract_query_info(query)
            # Add the special boost information
            query_info["special_boost"] = special_boost
            return query_info

        
        # Extract and parse JSON
        try:
            import json
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed_json = json.loads(json_match.group(0))
                return parsed_json
            else:
                # Fallback to simple extraction if JSON parsing fails
                return self._extract_query_info(query)
        except Exception as e:
            print(f"Error parsing DeepSeek response: {e}")
            return self._extract_query_info(query)
    
    def _extract_query_info(self, query: str) -> Dict:
        """
        Fallback method to extract query information using rules.
        
        Args:
            query: The search query text
            
        Returns:
            Dictionary with extracted information
        """
        query_lower = query.lower()
        
        # Extract product type
        product_type = ""
        if any(word in query_lower for word in ["cable", "charger", "cord"]):
            product_type = "cable"
        elif any(word in query_lower for word in ["headset", "headphone", "earphone", "earbud"]):
            product_type = "headphone"
            
        # Extract key features
        key_features = []
        if "quality" in query_lower:
            key_features.append("high quality")
        if "fast" in query_lower and "charging" in query_lower:
            key_features.append("fast charging")
        if "noise" in query_lower and "cancelling" in query_lower:
            key_features.append("noise cancelling")
        if "warranty" in query_lower:
            key_features.append("warranty")
            
        # Enhanced warranty detection
        if any(term in query_lower for term in ["warranty", "guarantee", "year"]):
            key_features.append("warranty")
        
        # Check for brand mentions
        if "iphone" in query_lower:
            key_features.append("iphone compatible")
            
        # Extract price constraint
        price_constraint = None
        price_match = re.search(r'under (\d+(\.\d+)?)\s*USD', query_lower)
        if price_match:
            price_constraint = float(price_match.group(1))
            
        return {
            "product_type": product_type,
            "key_features": key_features,
            "price_constraint": price_constraint
        }
    

    def _parse_deepseek_response(self, response, default_value=None):
        """
        Robustly parse DeepSeek responses with multiple fallback mechanisms.
        
        Args:
            response: Raw response from DeepSeek model
            default_value: Default value to return if parsing fails
            
        Returns:
            Parsed data or default value if parsing fails
        """
        import json
        import re
        
        # Early exit for empty responses
        if not response or not isinstance(response, str):
            return default_value
        
        # Try to parse as direct JSON first (best case)
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            # Not direct JSON, try to extract JSON block
            pass
        
        # Pattern for finding JSON-like object between curly braces
        #pattern = r'\{(?:[^{}]|(?R))*\}'
        #json_matches = re.findall(pattern, response, re.DOTALL)

        # Use a simpler non-recursive pattern to find JSON objects
        json_matches = re.findall(r'\{[^{]*?\}', response, re.DOTALL)
        
        # Try each potential JSON block found
        for potential_json in json_matches:
            try:
                # Clean up the JSON string before parsing
                cleaned_json = self._clean_json_string(potential_json)
                return json.loads(cleaned_json)
            except json.JSONDecodeError:
                continue
        
        # If we're looking for a rating/score and regex-based extraction
        num_pattern = r'(\d+(?:\.\d+)?)'
        if 'rate' in response.lower() or 'score' in response.lower():
            num_match = re.search(num_pattern, response)
            if num_match:
                try:
                    return {"score": float(num_match.group(1))}
                except (ValueError, TypeError):
                    pass
        
        # Last resort: simple keyword-based extraction for specific fields
        result = {}
        
        # Try to extract product_type
        product_type_match = re.search(r'product[_\s]type[\s\:\"\']+([\w\s]+)', response, re.IGNORECASE)
        if product_type_match:
            result["product_type"] = product_type_match.group(1).strip()
        
        # Try to extract features as a list
        features_section = re.search(r'features[\s\:\"\']+\[(.*?)\]', response, re.DOTALL | re.IGNORECASE)
        if features_section:
            features_text = features_section.group(1)
            features = re.findall(r'[\"\']([^\"\',]+)[\"\']', features_text)
            if features:
                result["key_features"] = features
        
        # If we found any fields this way, return them
        if result:
            return result
        
        # If all else fails, return the default value
        return default_value

    def _clean_json_string(self, json_str):
        """
        Clean and fix common JSON formatting issues.
        
        Args:
            json_str: Potentially malformed JSON string
            
        Returns:
            Cleaned JSON string
        """
        import re
        
        # Remove any trailing commas before closing braces/brackets
        cleaned = re.sub(r',\s*}', '}', json_str)
        cleaned = re.sub(r',\s*]', ']', cleaned)
        
        # Fix missing quotes around keys
        cleaned = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', cleaned)
        
        # Fix unquoted string values
        # This is tricky and might cause issues, so commenting it out for now
        # cleaned = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9_\s]+)(\,|})', r':"\1"\2', cleaned)
        
        # Fix Python True/False/None to JSON true/false/null
        cleaned = re.sub(r':\s*True', r':true', cleaned)
        cleaned = re.sub(r':\s*False', r':false', cleaned)
        cleaned = re.sub(r':\s*None', r':null', cleaned)
        
        return cleaned

    def analyze_with_prompt(self, prompt: str) -> Dict:
        """
        Use DeepSeek to analyze any custom prompt and return structured data.
        Enhanced version with robust error handling.
        
        Args:
            prompt: The prompt to analyze
            
        Returns:
            Structured data extracted from DeepSeek's response
        """
        model, tokenizer = self.load_model()
        
        # Enhance the prompt to encourage proper JSON formatting
        json_format_note = """
        FORMAT INSTRUCTIONS:
        - Return valid JSON only
        - Use double quotes for keys and string values
        - Do not add any text before or after the JSON object
        - For ratings, use numbers not strings
        """
        
        enhanced_prompt = prompt + "\n" + json_format_note
        
        inputs = tokenizer(enhanced_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                top_p=0.95,
                do_sample=True  # Set to True to make temperature and top_p work
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Use our enhanced parsing function
        parsed_response = self._parse_deepseek_response(response, default_value={"error": "Failed to parse response"})
        
        # If we couldn't parse the response at all, include the raw response for debugging
        if "error" in parsed_response:
            parsed_response["raw_response"] = response
        
        return parsed_response
    

    def rerank_results(self, 
                       query: str, 
                       results: pd.DataFrame, 
                       query_analysis: Dict = None,
                       top_n: int = 20) -> pd.DataFrame:
        """
        Rerank search results using DeepSeek's reasoning capabilities.
        
        Args:
            query: Original search query
            results: DataFrame containing search results
            query_analysis: Optional pre-analyzed query information
            top_n: Maximum number of results to rerank
            
        Returns:
            DataFrame with reranked results
        """
        if len(results) == 0:
            return results
            
        # Limit to top_n results for efficiency
        results_to_rerank = results.head(top_n).copy()
        
        # Get query analysis if not provided
        if query_analysis is None:
            query_analysis = self.analyze_query(query)
            
        model, tokenizer = self.load_model()
        
        # Initialize scores list
        semantic_scores = []
        
        # Add image feature analysis to prompt if available
        for _, row in results_to_rerank.iterrows():
            # Format product information
            product_desc = f"""
            Name: {row['product_name']}
            Category: {row['category']}
            Price: {row['price_usd']} USD
            """
            
            # Add product description
            if 'about_product' in row and not pd.isna(row['about_product']):
                product_desc += f"Description: {row['about_product'][:300]}...\n"
            
            # Add visual features if available
            if 'visual_features' in row and pd.notna(row['visual_features']):
                visual_features = row['visual_features']
                if isinstance(visual_features, dict) and visual_features:
                    product_desc += "Visual Features:\n"
                    if 'colors' in visual_features:
                        product_desc += f"- Colors: {', '.join(visual_features['colors'])}\n"
                    if 'design_elements' in visual_features:
                        product_desc += f"- Design: {', '.join(visual_features['design_elements'])}\n"
                    if 'quality_indicators' in visual_features:
                        product_desc += f"- Quality: {visual_features['quality_indicators']}\n"
            
            # Create prompt with query context
            prompt = f"""
            Rate how relevant this product is to the search query on a scale from 0 to 10.
            
            Search Query: "{query}"
            
            Product Information:
            {product_desc}
            
            Key search requirements:
            - Product Type: {query_analysis.get('product_type', 'Any')}
            - Important Features: {', '.join(query_analysis.get('key_features', []))}
            - Price Constraint: {query_analysis.get('price_constraint', 'None')}
            
            Consider exact category matches, feature matches, and price constraints.
            Output only the numerical score (0-10).
            """
            
            # Generate rating
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.2,
                    do_sample=True
                )
            
            # Extract numerical rating
            raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            try:
                match = re.search(r'(\d+(\.\d+)?)', raw_output)
                score = float(match.group(1)) if match else 0.0
                score = min(max(score, 0.0), 10.0)  # Ensure in range 0-10
            except Exception:
                score = 0.0
                
            semantic_scores.append(score)
        
        # Add semantic scores to results
        results_to_rerank['semantic_score'] = semantic_scores
        
        # Create a deep copy to ensure values don't get overwritten
        reranked_results = results_to_rerank.copy(deep=True)
        
        # Normalize original scores to 0-10 range
        if 'hybrid_score' in reranked_results.columns and len(reranked_results) > 1:
            min_score = reranked_results['hybrid_score'].min()
            max_score = reranked_results['hybrid_score'].max()
            if max_score > min_score:
                reranked_results['normalized_score'] = (
                    (reranked_results['hybrid_score'] - min_score) / 
                    (max_score - min_score) * 10
                )
            else:
                reranked_results['normalized_score'] = reranked_results['hybrid_score']
        else:
            # Fallback if hybrid_score isn't available
            reranked_results['normalized_score'] = reranked_results.get('hybrid_score', 0)
        
        # Calculate final score (weighted combination)
        reranked_results['final_score'] = (
            reranked_results['normalized_score'].fillna(0) * 0.3 + 
            reranked_results['semantic_score'].fillna(0) * 0.7
        )
        
        # Sort by final score
        reranked_results = reranked_results.sort_values('final_score', ascending=False)
        
         # Debug info
        if 'B009LJ2BXA' in reranked_results['product_id'].values:
            idx = reranked_results[reranked_results['product_id'] == 'B009LJ2BXA'].index[0]
            print(f"B009LJ2BXA semantic score: {reranked_results.at[idx, 'semantic_score']}")
            print(f"B009LJ2BXA final score: {reranked_results.at[idx, 'final_score']}")
            print(f"B009LJ2BXA rank after reranking: {reranked_results.index.get_loc(idx) + 1}")
        
        return reranked_results
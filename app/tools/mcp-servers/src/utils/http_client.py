"""
HTTP client utilities for making API requests.
"""
import httpx
from typing import Any, Dict, Optional
from src.config import settings
from src.utils.auth import auth_manager


class APIClient:
    """
    HTTP client for making requests to the telemedicine API.
    
    Handles:
    - Request/response formatting
    - Authentication
    - Error handling
    - Common headers
    """

    def __init__(self) -> None:
        self.base_url = settings.base_url
        self.timeout = settings.timeout

    def _get_headers(self, additional_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Build headers for API requests.
        
        Args:
            additional_headers: Optional additional headers to include
            
        Returns:
            Complete headers dictionary
        """
        headers = settings.get_headers()
        
        # Add authentication headers
        auth_headers = auth_manager.get_auth_headers()
        headers.update(auth_headers)
        
        # Add any additional headers
        if additional_headers:
            headers.update(additional_headers)
        
        return headers

    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Make a GET request.
        
        Args:
            endpoint: API endpoint (will be appended to base_url)
            params: Query parameters
            headers: Additional headers
            
        Returns:
            JSON response as dictionary
            
        Raises:
            httpx.HTTPError: If the request fails
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        request_headers = self._get_headers(headers)
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, params=params, headers=request_headers)
            response.raise_for_status()
            return response.json()

    async def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Make a POST request.
        
        Args:
            endpoint: API endpoint (will be appended to base_url)
            data: JSON body data
            params: Query parameters
            headers: Additional headers
            
        Returns:
            JSON response as dictionary
            
        Raises:
            httpx.HTTPError: If the request fails
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        request_headers = self._get_headers(headers)
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                url, json=data, params=params, headers=request_headers
            )
            response.raise_for_status()
            return response.json()

    async def put(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Make a PUT request (placeholder for future use).
        
        Args:
            endpoint: API endpoint
            data: JSON body data
            params: Query parameters
            headers: Additional headers
            
        Returns:
            JSON response as dictionary
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        request_headers = self._get_headers(headers)
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.put(
                url, json=data, params=params, headers=request_headers
            )
            response.raise_for_status()
            return response.json()

    async def delete(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Make a DELETE request (placeholder for future use).
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            headers: Additional headers
            
        Returns:
            JSON response as dictionary
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        request_headers = self._get_headers(headers)
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.delete(url, params=params, headers=request_headers)
            response.raise_for_status()
            return response.json()


# Global API client instance
api_client = APIClient()
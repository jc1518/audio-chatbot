from wordpress_xmlrpc import Client, WordPressPost
from wordpress_xmlrpc.methods.posts import NewPost
from wordpress_xmlrpc.exceptions import InvalidCredentialsError


class WordPressBlogger:
    """
    A class to handle posting content to WordPress blogs using XML-RPC.
    """

    def __init__(self, url, username, password):
        """
        Initialize WordPress connection.

        Args:
            url (str): WordPress site URL with /xmlrpc.php (e.g., 'https://example.com/xmlrpc.php')
            username (str): WordPress username
            password (str): WordPress password or application password
        """
        self.url = url
        self.username = username
        self.password = password
        self.client = None

    def connect(self):
        """
        Establish connection to WordPress site.

        Returns:
            bool: True if connection successful, False otherwise

        Raises:
            InvalidCredentialsError: If authentication fails
            ConnectionError: If cannot connect to the WordPress site
        """
        try:
            self.client = Client(self.url, self.username, self.password)
            return True
        except InvalidCredentialsError:
            raise InvalidCredentialsError(
                "Failed to authenticate with WordPress. Check credentials."
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to WordPress site: {str(e)}")

    def post_content(self, title, content, status="draft", categories=None, tags=None):
        """
        Post content to WordPress blog.

        Args:
            title (str): Title of the blog post
            content (str): Main content/body of the blog post
            status (str, optional): Post status ('draft' or 'publish'). Defaults to 'draft'.
            categories (list, optional): List of category names. Defaults to None.
            tags (list, optional): List of tags. Defaults to None.

        Returns:
            str: ID of the created post if successful

        Raises:
            ConnectionError: If not connected to WordPress
            ValueError: If invalid parameters provided
        """
        if not self.client:
            raise ConnectionError("Not connected to WordPress. Call connect() first.")

        if not title or not content:
            raise ValueError("Title and content are required.")

        post = WordPressPost()
        post.title = title
        post.content = content
        post.post_status = status

        if categories:
            post.terms_names = {"category": categories}
        if tags:
            post.terms_names = {"post_tag": tags}

        try:
            post_id = self.client.call(NewPost(post))
            return post_id
        except Exception as e:
            raise Exception(f"Failed to create post: {str(e)}")


# Example usage:
"""
# Initialize WordPress connection
wp = WordPressBlogger(
    url='https://example.com/xmlrpc.php',
    username='your_username',
    password='your_password'
)

# Connect to WordPress
wp.connect()

# Create a post
post_id = wp.post_content(
    title='My Generated Blog Post',
    content='This is the content of my blog post.',
    status='draft',  # or 'publish'
    categories=['AI', 'Technology'],
    tags=['ai', 'tech', 'blog']
)
"""

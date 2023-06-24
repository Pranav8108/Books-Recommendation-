from app import recommend


def test_recommend():
    result = recommend('The Da Vinci Code')
    assert len(result['recommended_books']) == 4
    assert all(book['Book-Title'] != 'The Da Vinci Code' for book in result['recommended_books'])
    
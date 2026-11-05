import asyncio
from app.database import client


async def check_mongodb():
    try:
        result = await client.admin.command("ping")
        print("MONGODB CONNECTION SUCCESSFUL")
        print(result)
    except Exception as error:
        print("MONGODB CONNECTION FAILED")
        print(type(error).__name__, str(error))
    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(check_mongodb())
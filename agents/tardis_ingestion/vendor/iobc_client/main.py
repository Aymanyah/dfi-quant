import sys
import os
import asyncio
from pathlib import Path
import platform

# tardis_client-$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)
# 自动根据当前系统读取不同的架构的目录
system = platform.system().lower()
machine = platform.machine()
runtime_dir = f"tardis_client-{system}-{machine}"

runtime = Path(__file__).parent.joinpath(runtime_dir, 'pyarmor_runtime_000000')
if runtime.exists():
    sys.path.insert(0, str(runtime.parent))
    os.environ['PYARMOR_RUNTIME'] = str(runtime)
else:
    raise Exception(f"not found tardis client runtime for {system} {machine}")


from tardis_client.tardis_client import TardisClient, Channel


TOKEN = os.environ["IOBC_DATA_TOKEN"]
async def example_usage():
    client = TardisClient(token=TOKEN)
    messages = client.replay(
        exchange="bitmex",
        from_date="2019-06-01",
        to_date="2019-06-02",
        filters=[Channel(name="trade", symbols=["XBTUSD","ETHUSD"]), Channel("orderBookL2", ["XBTUSD"])],
    )
    async for local_timestamp, message in messages:
        print(message)


if __name__ == "__main__":
    asyncio.run(example_usage())

# CDS (Cosmos Dataset Search) Client CLI

## Setup

The client CLI is the most convenient way to add video data or precomputed embedding parquet to a running Cosmos Dataset Search (CDS) service endpoint.

### Installation

Install the CDS CLI from the repository using the make target:

```bash
# From the repository root
make install-cds-cli
```

This will install the `cds` command along with its required dependencies (Ray, Fire, Rich, and python-magic) as optional dependencies. These packages are only installed when you specifically install the client, keeping the main project dependencies lean.

#### Manual Installation

Alternatively, you can install manually:

```bash
# Activate your virtual environment first
source .venv/bin/activate

# Install with client dependencies
pip install -e ".[client]"
```

### Activating the Environment

**Important:** The `cds` command is only available when the virtual environment is activated:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Verify the installation
cds --help
```

You'll need to activate the virtual environment in each new terminal session where you want to use the `cds` command.

### Interactive Configuration

Set up the CLI interactively to configure your API endpoint:

```bash
cds config set
```

This creates a configuration file at `~/.config/cds/config` with your API endpoint settings.

You may want to use the client with multiple endpoints. You can configure additional profiles using the `--profile` flag:

```bash
# Configure the default profile
cds config set

# Configure a local profile
cds config set --profile local

# Configure a production profile
cds config set --profile production
```

This creates a configuration file at `~/.config/cds/config` with multiple profiles:

```ini
[default]
api_endpoint = http://a26261c63d9e04874bb32e1809603c3d-108487706.us-west-2.elb.amazonaws.com

[local]
api_endpoint = http://localhost:8888

[production]
api_endpoint = http://production-endpoint.example.com
```

You can also manually edit `~/.config/cds/config` to add or modify profiles.

## Quick start

The commands will use the default profile, but you can override with `--profile ...`.
For complete documentation, run `cds --help` (artifact install) or `python3 cvds/src/visual_search/client/client.py --help` (local).

### List

To see all the collections in the database:

```bash
cds collections list
```

### Search (quick start)

To search a given collection:

```bash
cds search --collection-ids $collection_uuid --text-query 'person crossing the street' --top-k 1
```

### List available pipelines

List pipelines that one can use for new collections.

```bash
cds pipelines list
```

### Create a new collection

```bash
cds collections create --pipeline simple_image_search_vastdb
```

This will create a new collection. The collection_id will be printed to the console. You can specify more complex configurations for the index as well.

### Ingest data into a collection (S3-compatible storage)

```bash
cds ingest files s3://bucket/prefix --collection-id $collection_uuid --extensions .mp4 --num-workers 1 --batch-size 10
```

#### S3 Configuration

The CLI supports ingesting files from S3-compatible storage systems using `s3://...` URIs. You have two options for authentication:

**Option 1: Using environment variables (no profile needed)**

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_ENDPOINT_URL=http://your-s3-endpoint:9000  # Required for S3-compatible storage
export AWS_DEFAULT_REGION=us-east-1  # Optional

cds ingest files s3://bucket/prefix --collection-id $collection_uuid --extensions .mp4
```

**Option 2: Using AWS profile (recommended for multiple S3 endpoints)**

When using `--s3-profile`, you need to configure **both** files:

1. **`~/.aws/credentials`** - Access credentials:
```ini
[cds-s3]
aws_access_key_id = test
aws_secret_access_key = test
```

2. **`~/.aws/config`** - Endpoint and region configuration:
```ini
[profile cds-s3]
endpoint_url = http://vast-s3:4566 # in case of docker deployment or add proper s3/minio endpoint url
region = us-east-1
```

Then use the profile:
```bash
cds ingest files s3://cosmos-test-bucket/videos/ \
  --collection-id $collection_uuid \
  --extensions .mp4 \
  --s3-profile cds-s3
```

**Important Notes:**
- For S3-compatible storage (MinIO, Vast S3, etc.), the `endpoint_url` configuration in `~/.aws/config` is **required**
- Use `s3://` URIs, not presigned URLs
- The profile name in credentials file is `[cds-s3]`, but in config file it's `[profile cds-s3]`
- Depending on your environment, `--metadata-json` and `--output-log` may be optional or required

You can also ingest pre-computed embeddings. We expect a directory of parquet files or a single parquet file.

```bash
cds ingest embeddings $PARQUET --collection-id $collection_uuid --metadata-cols a,b,c --embeddings-col embeddings --id-cols d
```

The parquet must contain a column with embeddings (an array/list of floats), and can also contain additional columns for metadata that will be associated to each embedding. You can customize the command like: `--metadata-cols a,b,c --embeddings-col embeddings --id-cols d`

## Detailed command reference

Below are the available commands with their options. Most commands accept `--profile PROFILE` to select a configured endpoint profile.

### Config

Interactive configuration of credentials and endpoints.

```bash
cds config set
```

### Pipelines

List available pipelines.

```bash
cds pipelines list [--verbose] [--profile PROFILE]
```

- **--verbose**: Print full pipeline configuration details.
- **--profile**: Use a named credentials profile (e.g., `default`, `local`).

### Collections

Create a collection (supports overriding index type):

```bash
cds collections create \
  --pipeline PIPELINE \
  [--collection-id UUID] \
  [--config-yaml PATH] \
  [--name NAME] \
  [--index-type TYPE] \
  [--profile PROFILE]
```

- **--pipeline**: Pipeline to use for the collection (e.g., embedding model/index).
- **--collection-id**: Optional UUID to set an explicit ID; auto-generated if omitted.
- **--config-yaml**: Path to YAML with index/collection metadata and tags.
- **--name**: Human-readable description/name for the collection.
- **--index-type**: Override index type (e.g., `GPU_CAGRA`, `IVF_SQ8`).
- **--profile**: Use a named credentials profile for the target endpoint.

List collections:

```bash
cds collections list [--profile PROFILE]
```

- **--profile**: Use a named credentials profile.

Get a collection:

```bash
cds collections get COLLECTION_ID [--profile PROFILE]
```

- **COLLECTION_ID**: UUID of the target collection.
- **--profile**: Use a named credentials profile.

Delete a collection (irreversible):

```bash
cds collections delete COLLECTION_ID [--profile PROFILE]
```

- **COLLECTION_ID**: UUID of the collection to delete (irreversible).
- **--profile**: Use a named credentials profile.

### Ingest

Upload files from S3-compatible storage:

```bash
cds ingest files s3://bucket/prefix \
  --collection-id ID \
  [--num-workers N] \
  [--batch-size N] \
  [--metadata-json PATH] \
  [--strip-directory-path true|false] \
  [--extensions .mp4] \
  [--limit N] \
  [--timeout SEC] \
  [--s3-profile NAME] \
  [--output-log PATH] \
  [--existence-check must|with_timeout|skip] \
  [--profile PROFILE]
```

- positional `s3://bucket/prefix`: S3-compatible path prefix containing files.
- **--collection-id**: Target collection UUID (required).
- **--num-workers**: Number of parallel upload workers (Ray actors).
- **--batch-size**: Number of files per JSON request.
- **--metadata-json**: JSON mapping filename -> metadata object.
- **--strip-directory-path**: Store filenames without the path prefix.
- **--extensions**: File extensions to include (e.g., `.mp4`).
- **--limit**: Maximum number of files to ingest.
- **--timeout**: Request timeout in seconds.
- **--s3-profile**: AWS profile name for remote listing/auth. When using this option, you must configure both `~/.aws/credentials` (for access keys) and `~/.aws/config` (for endpoint_url and region). See S3 Configuration section above for details.
- **--output-log**: CSV file to log per-request responses.
- **--existence-check**: `must`, `with_timeout`, or `skip` object existence check.
- **--profile**: Use a named credentials profile.

Notes:

- Only `s3://` paths are supported for file ingestion (not presigned URLs).
- Use `--extensions .mp4` for video files.
- For S3-compatible storage, see the S3 Configuration section above for required setup.

Upload precomputed embeddings from parquet (file or directory of parquet files):

```bash
cds ingest embeddings --parquet-dataset s3://bucket/path/to.parquet \
  --collection-id ID \
  --id-cols COL1[,COL2,...] \
  [--num-workers N] \
  [--embeddings-col NAME] \
  [--metadata-cols COLS] \
  [--fillna true|false] \
  [--limit N] \
  [--timeout SEC] \
  [--s3-profile NAME] \
  [--output-log PATH] \
  [--profile PROFILE]
```

- **--parquet-dataset**: Parquet file or directory with embeddings and metadata.
- **--collection-id**: Target collection UUID (required).
- **--id-cols**: Columns to hash into stable IDs (required).
- **--num-workers**: Number of parallel upload workers (Ray actors).
- **--embeddings-col**: Column containing embedding vectors (default: `embeddings`).
- **--metadata-cols**: Comma-separated metadata columns to include.
- **--fillna**: Fill NaNs in metadata columns (default: true).
- **--limit**: Maximum number of rows to ingest.
- **--timeout**: Request timeout in seconds.
- **--s3-profile**: AWS profile name for S3 access. When using this option, you must configure both `~/.aws/credentials` (for access keys) and `~/.aws/config` (for endpoint_url and region). See S3 Configuration section above for details.
- **--output-log**: CSV file to log per-request responses.
- **--profile**: Use a named credentials profile.

### Search

Text search across one or more collections:

```bash
cds search \
  --collection-ids ID[,ID...] \
  --text-query TEXT \
  [--top-k K] \
  [--generate-asset-url true|false] \
  [--profile PROFILE]
```

- **--collection-ids**: One or more collection UUIDs (comma-separated).
- **--text-query**: Text query to retrieve nearest items.
- **--top-k**: Number of results to return.
- **--generate-asset-url**: Request presigned asset URLs in results.
- **--profile**: Use a named credentials profile.

### Secrets

Create/update secrets:

```bash
cds secrets set -n NAME -k key1,key2 -v val1,val2 [--profile PROFILE]
```

- **-n, --name**: Secrets namespace/name to create or update.
- **-k, --keys**: Comma-separated keys.
- **-v, --values**: Comma-separated values (one per key).
- **--profile**: Use a named credentials profile.

Delete secrets:

```bash
cds secrets delete -n NAME [--profile PROFILE]
```

- **-n, --name**: Secrets namespace/name to delete.
- **--profile**: Use a named credentials profile.

List secrets:

```bash
cds secrets list [--profile PROFILE]
```

- **--profile**: Use a named credentials profile.

import json
import pandas as pd


def filter_comments_by_post_ids(id_file, comments_file, output_file):
    """
    Filter Reddit comments that belong to a list of post IDs and save only
    parent_id, comment_id, and body columns.
    """

    print(f"\nLoading post IDs from: {id_file}")
    post_ids = set(pd.read_csv(id_file)["post_id"].astype(str))
    print(f"Loaded {len(post_ids):,} post IDs")

    filtered = []

    print(f"Filtering comments from: {comments_file}")
    with open(comments_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                comment = json.loads(line)
                link_id = comment.get("link_id", "")
                comment_id = comment.get("id", "")
                body = comment.get("body", "")

                # Reddit comment 'link_id' format: "t3_<post_id>"
                post_id = link_id.replace("t3_", "")

                if post_id in post_ids:
                    filtered.append(
                        {
                            "parent_id": comment.get("parent_id", ""),
                            "comment_id": comment_id,
                            "body": body.strip(),
                        }
                    )

            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(filtered, columns=["parent_id", "comment_id", "body"])
    df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"Saved {len(df):,} filtered comments to: {output_file}")

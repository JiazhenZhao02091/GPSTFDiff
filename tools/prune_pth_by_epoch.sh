#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOF'
Usage:
  prune_pth_by_epoch.sh -d <DIR> -t <THRESHOLD> [--delete] [--prefix <regex>] [--verbose]

功能:
  递归删除 DIR 下所有 *.pth 中，文件名匹配 *_epoch_<数字>.pth 且 epoch < THRESHOLD 的文件。
  默认 dry-run 仅打印将要删除的文件；加 --delete 才真正删除。

参数:
  -d, --dir        目标目录（递归）
  -t, --threshold  epoch 阈值（删除 < 阈值）
  --delete         真正执行删除（危险）
  --prefix         可选：自定义匹配正则（Perl regex），默认: /_epoch_(\d+)\.pth$/
  --verbose        输出统计信息
  -h, --help       帮助

示例:
  # 预览（不会删除）
  ./prune_pth_by_epoch.sh -d /data/zhaojiazhen/STF/results/dinostf -t 900

  # 真删
  ./prune_pth_by_epoch.sh -d /data/zhaojiazhen/STF/results/dinostf -t 900 --delete
EOF
}

DIR=""
TH=""
DO_DELETE=0
VERBOSE=0
# 默认只匹配类似 model_epoch_999.pth（以 _epoch_<num>.pth 结尾）
REGEX='_epoch_(\d+)\.pth$'

# 解析参数
while [[ $# -gt 0 ]]; do
	case "$1" in
		-d|--dir) DIR="${2:-}"; shift 2;;
		-t|--threshold) TH="${2:-}"; shift 2;;
		--delete) DO_DELETE=1; shift;;
		--prefix) REGEX="${2:-}"; shift 2;;
		--verbose) VERBOSE=1; shift;;
		-h|--help) usage; exit 0;;
		*) echo "Unknown arg: $1" >&2; usage; exit 2;;
	esac
done

# 校验
if [[ -z "$DIR" || -z "$TH" ]]; then
	echo "Error: --dir and --threshold are required." >&2
	usage
	exit 2
fi
if [[ ! -d "$DIR" ]]; then
	echo "Error: DIR does not exist or is not a directory: $DIR" >&2
	exit 2
fi
if ! [[ "$TH" =~ ^[0-9]+$ ]]; then
	echo "Error: THRESHOLD must be a non-negative integer: $TH" >&2
	exit 2
fi

# 预览/删除逻辑
# 说明：用 -print0 / xargs -0 保证路径含空格也安全
# 只对匹配 REGEX 的 *.pth 生效；其他 *.pth 不动
if [[ $DO_DELETE -eq 0 ]]; then
	# dry-run：打印将要删除的文件（一行一个）
	find "$DIR" -type f -name '*.pth' -print0 \
		| perl -0ne '
			my $th = $ENV{TH};
			my $re = $ENV{RE};
			while (m{^(.+?)\0}sg) {
				my $f = $1;
				if ($f =~ /$re/ && $1 < $th) { print "$f\n"; }
				}
			' TH="$TH" RE="$REGEX"

			if [[ $VERBOSE -eq 1 ]]; then
				COUNT="$(find "$DIR" -type f -name '*.pth' -print0 \
					| perl -0ne '
									my $th = $ENV{TH};
									my $re = $ENV{RE};
									my $c = 0;
									while (m{^(.+?)\0}sg) {
										my $f = $1;
										if ($f =~ /$re/ && $1 < $th) { $c++; }
										}
									END { print $c; }
									' TH="$TH" RE="$REGEX")"
									echo "Dry-run: matched files to delete = $COUNT" >&2
			fi

		else
			# delete：真正删除
			if [[ $VERBOSE -eq 1 ]]; then
				echo "Deleting files under: $DIR (epoch < $TH, regex: /$REGEX/)" >&2
			fi

			find "$DIR" -type f -name '*.pth' -print0 \
				| perl -0ne '
							my $th = $ENV{TH};
							my $re = $ENV{RE};
							while (m{^(.+?)\0}sg) {
								my $f = $1;
								if ($f =~ /$re/ && $1 < $th) { print "$f\0"; }
								}
							' TH="$TH" RE="$REGEX" \
								| xargs -0 -r rm -f --

							if [[ $VERBOSE -eq 1 ]]; then
								echo "Done." >&2
							fi
fi


import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load_dataset(default_path: str) -> pd.DataFrame:
	"""Load dataset either from default path or from user upload."""
	st.sidebar.subheader("Data source")
	uploaded = st.sidebar.file_uploader("Upload a CSV (optional)", type=["csv"])
	if uploaded is not None:
		return pd.read_csv(uploaded)
	if os.path.exists(default_path):
		return pd.read_csv(default_path)
	raise FileNotFoundError(
		f"Could not find dataset at {default_path}. Upload a CSV in the sidebar."
	)


def guess_cluster_columns(df: pd.DataFrame) -> List[str]:
	"""Heuristically guess cluster label columns."""
	candidates = []
	lower_cols = {c.lower(): c for c in df.columns}
	for key in [
		"cluster",
		"label",
		"segment",
		"group",
		"cluster_id",
		"cluster_label",
		"kmeans",
		"assignment",
	]:
		if key in lower_cols:
			candidates.append(lower_cols[key])
	# Also include integer/categorical-like columns with few unique values
	for col in df.columns:
		unique_count = df[col].nunique(dropna=True)
		if 2 <= unique_count <= min(20, max(2, len(df) // 20)):
			candidates.append(col)
	# De-duplicate preserving order
	seen = set()
	ordered = []
	for c in candidates:
		if c not in seen:
			ordered.append(c)
			seen.add(c)
	return ordered


def select_features_for_projection(df: pd.DataFrame, cluster_col: Optional[str]) -> List[str]:
	"""Select numeric feature columns for dimensionality reduction."""
	numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
	if cluster_col in numeric_cols:
		numeric_cols.remove(cluster_col)
	return numeric_cols


def compute_projection(
	df: pd.DataFrame,
	feature_cols: List[str],
	method: str,
	random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Project high-dimensional features to 2D using PCA or t-SNE."""
	X = df[feature_cols].fillna(df[feature_cols].median()).to_numpy()
	if method == "PCA":
		pca = PCA(n_components=2, random_state=random_state)
		XY = pca.fit_transform(X)
		return XY[:, 0], XY[:, 1]
	elif method == "t-SNE":
		tsne = TSNE(n_components=2, random_state=random_state, init="pca", learning_rate="auto")
		XY = tsne.fit_transform(X)
		return XY[:, 0], XY[:, 1]
	else:
		raise ValueError("Unknown method: " + method)


def sidebar_controls(df: pd.DataFrame) -> dict:
	"""Render sidebar controls and return selections."""
	st.sidebar.title("Clustering Dashboard")
	st.sidebar.caption("Explore student clusters interactively")

	cluster_candidates = guess_cluster_columns(df)
	cluster_col = st.sidebar.selectbox(
		"Cluster label column",
		options=["<none>"] + cluster_candidates,
		index=1 if cluster_candidates else 0,
	)
	cluster_col = None if cluster_col == "<none>" else cluster_col

	projection_method = st.sidebar.selectbox("2D projection method", ["PCA", "t-SNE"], index=0)

	# Cluster filter
	cluster_filter = None
	if cluster_col is not None:
		values = df[cluster_col].dropna().unique()
		values = sorted(values.tolist())
		cluster_filter = st.sidebar.multiselect("Filter by cluster(s)", options=values, default=values)

	# Text search
	search_text = st.sidebar.text_input("Search text (matches any column)")

	# Feature selection for projection
	feature_cols_all = select_features_for_projection(df, cluster_col)
	default_feats = feature_cols_all[: min(20, len(feature_cols_all))]
	selected_features = st.sidebar.multiselect(
		"Features for projection",
		options=feature_cols_all,
		default=default_feats,
		help="Choose numeric features used for PCA/t-SNE",
	)

	return {
		"cluster_col": cluster_col,
		"cluster_filter": cluster_filter,
		"projection_method": projection_method,
		"selected_features": selected_features,
		"search_text": search_text,
	}


def room_allocation_controls(df: pd.DataFrame, cluster_col: Optional[str]) -> dict:
	"""Sidebar controls for dorm room allocation."""
	st.sidebar.markdown("---")
	st.sidebar.subheader("Dorm room allocation")
	capacity = st.sidebar.number_input("Room capacity", min_value=2, max_value=12, value=6, step=1)

	# Identify an ID column to label students
	id_candidates = [c for c in df.columns if df[c].is_unique]
	id_col = st.sidebar.selectbox(
		"Student identifier column",
		options=["<index>"] + id_candidates,
		index=0,
		help="Used to label students in room assignments",
	)
	id_col = None if id_col == "<index>" else id_col

	policy = st.sidebar.selectbox(
		"Allocation policy",
		options=[
			"Pack by cluster (maximize homogeneity)",
			"Round-robin across clusters (increase diversity)",
			"Random shuffle",
		],
		index=0,
	)

	random_state = st.sidebar.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)

	return {
		"capacity": int(capacity),
		"id_col": id_col,
		"policy": policy,
		"random_state": int(random_state),
		"cluster_col": cluster_col,
	}


def allocate_rooms(
	df: pd.DataFrame,
	cluster_col: str,
	id_col: Optional[str],
	capacity: int,
	policy: str,
	random_state: int = 42,
) -> pd.DataFrame:
	"""Return a DataFrame with room assignments: adds columns 'room_id' and 'bed'."""
	if cluster_col is None:
		raise ValueError("Select a cluster label column to allocate rooms.")
	if capacity <= 1:
		raise ValueError("Capacity must be at least 2.")

	work = df.copy().reset_index(drop=True)
	label_col = id_col if id_col is not None else None

	# Ordering according to policy
	if policy.startswith("Pack by cluster"):
		work = work.sort_values(by=[cluster_col]).reset_index(drop=True)
	elif policy.startswith("Round-robin"):
		np_random = np.random.RandomState(random_state)
		cluster_groups = {k: g.sample(frac=1.0, random_state=np_random.randint(0, 1_000_000)) for k, g in work.groupby(cluster_col, sort=False)}
		order = []
		while any(len(g) > 0 for g in cluster_groups.values()):
			for k in sorted(cluster_groups.keys(), key=lambda x: str(x)):
				g = cluster_groups[k]
				if len(g) == 0:
					continue
				order.append(g.iloc[0])
				cluster_groups[k] = g.iloc[1:]
		work = pd.DataFrame(order).reset_index(drop=True)
	elif policy.startswith("Random"):
		work = work.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
	else:
		raise ValueError("Unknown policy")

	# Assign rooms and bed numbers
	n = len(work)
	room_ids = [f"Room-{i+1}" for i in range((n + capacity - 1) // capacity)]
	room_col = []
	bed_col = []
	for idx in range(n):
		room_index = idx // capacity
		bed_index = idx % capacity + 1
		room_col.append(room_ids[room_index])
		bed_col.append(bed_index)

	work["room_id"] = room_col
	work["bed"] = bed_col

	# Keep identifier column if provided, else create an index-based id for display
	if label_col is None:
		work["student_id"] = work.index.astype(str)
	else:
		work["student_id"] = work[label_col].astype(str)

	return work


def render_room_visualizations(assigned: pd.DataFrame, cluster_col: str, capacity: int):
	st.subheader("Dorm rooms composition")
	# Stacked bar of cluster counts per room
	counts = assigned.groupby(["room_id", cluster_col]).size().reset_index(name="count")
	fig = px.bar(
		counts,
		x="room_id",
		y="count",
		color=cluster_col,
		title="Cluster composition per room",
		barmode="stack",
	)
	st.plotly_chart(fig, use_container_width=True)

	# Homogeneity metric per room: share of majority cluster
	room_stats = (
		assigned.groupby(["room_id", cluster_col]).size().reset_index(name="count")
		.sort_values(["room_id", "count"], ascending=[True, False])
	)
	majority = room_stats.groupby("room_id").agg({"count": "max"}).rename(columns={"count": "majority_count"})
	sizes = assigned.groupby("room_id").size().rename("room_size")
	merged = majority.join(sizes, how="inner")
	merged["homogeneity"] = (merged["majority_count"] / merged["room_size"]).astype(float)
	fig2 = px.histogram(merged.reset_index(), x="homogeneity", nbins=capacity + 2, title="Room homogeneity distribution (1.0 = all same cluster)")
	st.plotly_chart(fig2, use_container_width=True)

	with st.expander("Room assignments (table)"):
		st.dataframe(assigned[["room_id", "bed", "student_id", cluster_col]].sort_values(["room_id", "bed"]))

	# Download
	st.download_button(
		label="Download room assignments CSV",
		data=assigned[["room_id", "bed", "student_id", cluster_col]].to_csv(index=False).encode("utf-8"),
		file_name="room_assignments.csv",
		mime="text/csv",
	)


def apply_filters(df: pd.DataFrame, cluster_col: Optional[str], cluster_filter, search_text: str) -> pd.DataFrame:
	filtered = df.copy()
	if cluster_col is not None and cluster_filter is not None and len(cluster_filter) > 0:
		filtered = filtered[filtered[cluster_col].isin(cluster_filter)]
	if search_text:
		pattern = str(search_text).strip().lower()
		mask = pd.Series(False, index=filtered.index)
		for col in filtered.columns:
			col_vals = filtered[col].astype(str).str.lower().str.contains(pattern, na=False)
			mask = mask | col_vals
		filtered = filtered[mask]
	return filtered


def render_cluster_overview(df: pd.DataFrame, cluster_col: Optional[str]):
	st.subheader("Cluster overview")
	if cluster_col is None:
		st.info("Select a cluster label column in the sidebar to see cluster sizes.")
		return
	counts = df[cluster_col].value_counts(dropna=False).reset_index()
	counts.columns = [cluster_col, "count"]
	fig = px.bar(counts, x=cluster_col, y="count", color=cluster_col, title="Cluster sizes")
	st.plotly_chart(fig, use_container_width=True)


def render_projection(df: pd.DataFrame, cluster_col: Optional[str], method: str, feature_cols: List[str]):
	st.subheader("2D projection")
	if len(feature_cols) < 2:
		st.warning("Select at least 2 numeric features for projection.")
		return
	try:
		x, y = compute_projection(df, feature_cols, method)
		viz_df = df.copy()
		viz_df["x"] = x
		viz_df["y"] = y
		color = cluster_col if cluster_col is not None else None
		fig = px.scatter(
			viz_df,
			x="x",
			y="y",
			color=color,
			hover_data=[c for c in df.columns if c != cluster_col],
			title=f"{method} projection",
			opacity=0.85,
		)
		st.plotly_chart(fig, use_container_width=True)
	except Exception as e:
		st.error(f"Projection failed: {e}")


def render_feature_stats(df: pd.DataFrame, cluster_col: Optional[str]):
	st.subheader("Feature statistics")
	numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
	if not numeric_cols:
		st.info("No numeric columns to summarize.")
		return
	by_cluster = st.checkbox("Show by cluster", value=cluster_col is not None)
	if by_cluster and cluster_col is not None:
		desc = df.groupby(cluster_col)[numeric_cols].describe().transpose()
		st.dataframe(desc)
	else:
		st.dataframe(df[numeric_cols].describe().transpose())


def main():
	st.set_page_config(page_title="University Student Clusters", layout="wide")
	st.title("University Student Clusters Dashboard")
	st.caption("Interactively explore clustering results from the student dataset")

	default_csv = os.path.join(os.getcwd(), "EncodedWomen_english.csv")
	df = load_dataset(default_csv)

	controls = sidebar_controls(df)
	room_controls = room_allocation_controls(df, controls["cluster_col"])
	filtered_df = apply_filters(
		df,
		controls["cluster_col"],
		controls["cluster_filter"],
		controls["search_text"],
	)

	# KPIs
	col1, col2, col3 = st.columns(3)
	with col1:
		st.metric("Total records", f"{len(df):,}")
	with col2:
		st.metric("Filtered records", f"{len(filtered_df):,}")
	with col3:
		st.metric("Features (numeric)", f"{len(select_features_for_projection(df, controls['cluster_col']))}")

	# Visuals
	render_cluster_overview(filtered_df, controls["cluster_col"]) 
	render_projection(
		filtered_df,
		controls["cluster_col"],
		controls["projection_method"],
		controls["selected_features"],
	)
	render_feature_stats(filtered_df, controls["cluster_col"])

	# Room allocation section
	st.markdown("---")
	st.header("Dorm room allocation")
	if controls["cluster_col"] is None:
		st.info("Select a cluster label column in the sidebar to allocate rooms.")
	else:
		try:
			assigned = allocate_rooms(
				filtered_df,
				cluster_col=controls["cluster_col"],
				id_col=room_controls["id_col"],
				capacity=room_controls["capacity"],
				policy=room_controls["policy"],
				random_state=room_controls["random_state"],
			)
			st.caption(
				f"Assigned {len(assigned)} students to {(len(assigned) + room_controls['capacity'] - 1) // room_controls['capacity']} rooms (capacity {room_controls['capacity']})."
			)
			render_room_visualizations(assigned, controls["cluster_col"], room_controls["capacity"])
		except Exception as e:
			st.error(f"Room allocation failed: {e}")

	# Data preview and download
	st.subheader("Data preview")
	st.dataframe(filtered_df.head(1000))
	st.download_button(
		label="Download filtered CSV",
		data=filtered_df.to_csv(index=False).encode("utf-8"),
		file_name="filtered_students.csv",
		mime="text/csv",
	)


if __name__ == "__main__":
	main()



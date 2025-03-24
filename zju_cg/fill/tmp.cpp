// 4-1计算交点并填充
void FillPolygon(HDC hdc) {
    vector<Edge> edges;

    // 创建边表
    int n = polygon1.size();
    for (int i = 0; i < n; ++i) {
        POINT p1 = polygon1[i];
        POINT p2 = polygon1[(i + 1) % n];

        if (p1.y > p2.y) swap(p1, p2);
        if (p1.y != p2.y) {
            Edge edge;
            edge.ymin = p1.y;
            edge.ymax = p2.y;
            edge.x = p1.x;
            edge.slope = static_cast<float>(p2.x - p1.x) / (p2.y - p1.y);
            edges.push_back(edge);
        }
    }

    // 对边表按 y 坐标排序
    sort(edges.begin(), edges.end(), [](Edge a, Edge b) {
        return a.ymin < b.ymin;
        });

    // 扫描线填充
    for (int y = edges.front().ymin; y <= edges.back().ymax; ++y) {
        vector<float> intersections;

        // 查找当前扫描线与边的交点
        for (const Edge& edge : edges) {
            if (edge.ymin == y) {
                intersections.push_back(edge.x);
            }
            else if (y > edge.ymin && y < edge.ymax) {
                // 计算交点
                float x = edge.x + edge.slope * (y - edge.ymin);
                intersections.push_back(x);
            }
        }

        // 排序交点
        sort(intersections.begin(), intersections.end());

        // 填充交点之间的区域
        for (size_t i = 0; i < intersections.size(); i += 2) {
            if (i + 1 < intersections.size()) {
                HBRUSH hBrush = CreateSolidBrush(RGB(255, 0, 0)); // 红色
                SelectObject(hdc, hBrush);
                Rectangle(hdc, static_cast<int>(intersections[i]), y, static_cast<int>(intersections[i + 1]), y + 1);
                DeleteObject(hBrush);
            }
        }
    }
}

// 4-2获取边列表
vector<Edge> getEdges(const vector<POINT>& polygon) {
    vector<Edge> edges;
    int numVertices = polygon.size();
    for (int i = 0; i < numVertices; i++) {
        POINT p1 = polygon[i];
        POINT p2 = polygon[(i + 1) % numVertices];
        if (p1.y > p2.y) {
            swap(p1, p2); // 确保 p1 是较低的点
        }
        if (p1.y != p2.y) { // 排除水平边
            Edge edge;
            edge.ymin = p1.y;
            edge.ymax = p2.y;
            edge.x = p1.x;
            edge.slope = (float)(p2.x - p1.x) / (p2.y - p1.y);
            edges.push_back(edge);
        }
    }
    return edges;
}

// 种子填充算法
void seedFill(HDC hdc, int x, int y, COLORREF fillColor) {
    COLORREF targetColor = GetPixel(hdc, x, y);
    if (targetColor == fillColor) return; // 已经是目标颜色，退出

    vector<POINT> stack;
    stack.push_back({ x, y });

    while (!stack.empty()) {
        POINT p = stack.back();
        stack.pop_back();

        if (GetPixel(hdc, p.x, p.y) != targetColor) {
            SetPixel(hdc, p.x, p.y, fillColor);

            // 将相邻像素加入栈
            stack.push_back({ p.x + 1, p.y });
            stack.push_back({ p.x - 1, p.y });
            stack.push_back({ p.x, p.y + 1 });
            stack.push_back({ p.x, p.y - 1 });
        }
    }
}


VOID SeedFill(HDC hdc, int x, int y, COLORREF c, COLORREF d, int direction) {
	//c: background color
	//d: fill color
	if (GetPixel(hdc, x, y) == c) {
		SetPixel(hdc, x, y, d);
		if (direction != 2)
			SeedFill(hdc, x - 1, y, c, d, 1);
		if (direction != 1)
			SeedFill(hdc, x + 1, y, c, d, 2);
		if (direction != 4)
			SeedFill(hdc, x, y - 1, c, d, 3);
		if (direction != 3)
			SeedFill(hdc, x, y + 1, c, d, 4);
	}
}

typedef struct AET
{
	float x;
	float dx;
	int ymax;
	struct AET *next;
}Aet;

static Aet *edges[500], *active;

int yNext(int k, int num, POINT *ps)
{
	int j;
	if (k > num - 2)
		j = 0;
	else j = k + 1;
	while (ps[j].y == ps[k].y) {
		if (j > num - 2)
			j = 0;
		else j++;
	}
	return (ps[j].y);
}

void InsertEdge(Aet *list, Aet *edge)
{
	Aet *p, *q = list;
	p = q->next;
	while (p != NULL)
	{
		if (edge->x < p->x)
			p = NULL;
		else
		{
			q = p;
			p = p->next;
		}
	}
	edge->next = q->next;
	q->next = edge;
}

void MakeEdge(POINT low, POINT up, int yp, Aet *edge, Aet *edges[])
{
	edge->dx = (float)(up.x - low.x) / (up.y - low.y);
	edge->x = low.x;
	if (up.y < yp)
		edge->ymax = up.y - 1;
	else edge->ymax = up.y;
	InsertEdge(edges[low.y], edge);
}

void EdgeList(POINT *ps, int num, Aet *edges[])
{
	Aet *edge;
	POINT P1, P2;
	int i, yp = ps[num - 2].y;
	P1.x = ps[num - 1].x;
	P1.y = ps[num - 1].y;
	for (i = 0; i < num; ++i)
	{
		P2 = ps[i];
		if (P1.y != P2.y)
		{
			edge = (Aet *)malloc(sizeof(Aet));
			edge = (Aet *)malloc(sizeof(Aet));

			if (P1.y < P2.y)
				MakeEdge(P1, P2, yNext(i, num, ps), edge, edges);
			else
				MakeEdge(P2, P1, yp, edge, edges);
		}
		yp = P1.y;
		P1 = P2;
	}
}

void ActiveList(int scan, Aet *active, Aet *edge[])
{
	Aet *p, *q;
	p = edges[scan]->next;
	while (p)
	{
		q = p->next;
		InsertEdge(active, p);
		p = q;
	}
}

void ScanFill(int scan, Aet *active, HDC hdc)
{
	Aet *p, *q;
	int i;
	p = active->next;
	while (p != NULL)
	{
		q = p->next;
		for (i = p->x; i < q->x; i++)
			SetPixel(hdc, i, scan, RGB(90, 130, 50));
		p = q->next;
	}
}

void Delete(Aet *q)
{
	Aet *p = q->next;
	q->next = p->next;
	free(p);
}

void UpdateActiveList(int scan, Aet *active)
{
	Aet *q = active, *p = active->next;
	while (p)
	{
		if (scan >= p->ymax)
		{
			p = p->next;
			Delete(q);
		}
		else
		{
			p->x = p->x + p->dx;
			q = p;
			p = p->next;
		}
	}
}

void ReActiveList(Aet *active)
{
	Aet *q, *p = active->next;
	active->next = NULL;
	while (p)
	{
		q = p->next;
		InsertEdge(active, p);
		p = q;
	}
}

void ScanLineFillPolygon(POINT *ps, int num, HDC hdc)
{
	int i, scan, scanmax = 0, scanmin = 500;

	for (i = 0; i < num; i++)
	{
		if (ps[i].y > scanmax)
			scanmax = ps[i].y;
		if (ps[i].y < scanmin)
			scanmin = ps[i].y;
	}

	for (scan = scanmin; scan < scanmax; scan++)
	{
		edges[scan] = (Aet *)malloc(sizeof(Aet));
		edges[scan]->next = NULL;
	}
	EdgeList(ps, num, edges);
	active = (Aet *)malloc(sizeof(Aet));
	active->next = NULL;

	for (scan = scanmin; scan < scanmax; scan++)
	{
		ActiveList(scan, active, edges);
		if (active->next)
		{
			ScanFill(scan, active, hdc);
			UpdateActiveList(scan, active);
			ReActiveList(active);
		}
	}
}
# MOBA
# 图论

## 拓扑排序

```cpp
queue<int>q; // 如果题目有偏序条件，可以改为优先队列
vector<int>a;
vector<int>e[N];
void topsort() {
    for (int i = 1; i <= n; i++) {
        // 入度为0的节点入队
        if (!p[i]) { 
            q.push(i);
        }
    }
    while (!q.empty()) {
        int t = q.front();
        q.pop();
        a.push_back(t);
        for (auto &i : e[t]) {
            p[i]--;
            if (!p[i]) {
                q.push(i);
            }
        } 
    }
}
```

## 最小生成树

```cpp
int n, m; // n 为节点数， m 为边数

struct node {
    int a;
    int b;
    int w;
}e[M];

bool cmp(node a, node b) {
    return a.w < b.w;
}

int kruskal() {
    int res = 0, cnt = 0;
    for (int i = 0; i <= n; i++) p[i] = i;
    sort(e + 1, e + 1 + m; cmp);
    for (int i = 1; i <= m; i++) {
        int a = e[i].a, b = e[i].b, w = e[i].w;
        if (find(a) != find(b)) {
            join(a, b);
            res += w;
            cnt++;
        }
    }
    return res;
}
```
## 并查集

```cpp
// 初始化操作
for (int i = 0; i <= n; i++) {
    p[i] = i;
}

// 查找，路径压缩
int find(int x) {
    if (x == p[x]) return x;
    return p[x] = find(p[x]);
}

// 合并操作
void join(int a, int b) {
    int aa = find(a), bb = find(b);
    if (aa == bb) return;
    p[aa] = bb;
}
```

## 单源最短路 dij 堆优化

```cpp
vector<node>e[N];
void dij(int s) {
    priority_queue< PII, vector<PII>, greater<PII> >q;
    for (int i = 0; i <= n + 1; i++) {
        d[i] = INF;
        vis[i] = false; 
    }
    d[s] = 0;
    q.push({0, s});
    while (!q.empty()) {
        PII t = q.top();
        q.pop();
        int u = q.second, dis = q.first;
        if (vis[u]) continue;
        vis[u] = true;
        for (auto &i : e[u]) {
            if (d[i.v] > d[u] + i.w) {
                d[i.v] = d[u] + i.w;
                q.push({d[i.v], i.v});
            }
        } 
    } 
} 
```

## 多源最短路 费得比尔曼算法

```cpp
struct Edge
{
    int a, b, c;
}edges[M];

int n, m;
int dist[N];
int last[N];

void bellman_ford()
{
    memset(dist, 0x3f, sizeof dist);

    dist[1] = 0;
    for (int i = 0; i < n; i ++ )
    {
        memcpy(last, dist, sizeof dist);
        for (int j = 0; j < m; j ++ )
        {
            auto e = edges[j];
            dist[e.b] = min(dist[e.b], last[e.a] + e.c);
        }
    }
}
```
## 多源最短路 spfa算法

```cpp
vector<node>e[N];
int spfa() {
    memset(dist, 0x3f, sizeof dist);
    d[1] = 0;
    queue<int> q;
    q.push(1);
    st[1] = true;
    while (!q.empty()) {
        int t = q.front();
        q.pop();
        st[t] = false;
        for (auto &i : e[t]) {
            if (d[i.v] > d[t] + i.w) {
                d[i.v] = d[t] + i.w;
                if (!st[i.v]) {
                    q.push(i.v);
                    st[i.v] = true;
                }
            }
        }
    }
    if (d[n] == 0x3f3f3f3f) return -1;
    return d[n];
}
```
## 判负环 spfa算法
```cpp
bool fspfa() {
    // 不需要初始化dist数组
    // 原理：如果某条最短路径上有n个点（除了自己），那么加上自己之后一共有n+1个点，由抽屉原理一定有两个点相同，所以存在环。
    queue<int> q;
    for (int i = 1; i <= n; i ++ ) {
        q.push(i);
        st[i] = true;
    }
    while (!q.empty()) {
        int t = q.front();
        q.pop();
        st[t] = false;
        for (auto &i : e[t]) {
            if (d[i.v] > d[t] + i.w) {
                d[i.v] = d[t] + i.w;
                cnt[i.v] = cnt[t] + 1;
                if (cnt[i.v] >= n) return true;       // 如果从1号点到x的最短路中包含至少n个点（不包括自己），则说明存在环
                if (!st[i.v]) {
                    q.push(i.v);
                    st[i.v] = true;
                }
            }
        }
    }
    return false;
}
```

## 全源最短路  floyd算法
```cpp
//初始化：
for (int i = 1; i <= n; i ++ )
    for (int j = 1; j <= n; j ++ )
        if (i == j) d[i][j] = 0;
        else d[i][j] = INF;

void floyd() {
    for (int k = 1; k <= n; k++) {
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                d[i][j] = min(d[i][j], d[i][k] + d[k][j]);
            }
        }
    }
}
```

## 全源最短路 Johnson

```cpp
struct edge{
    int v,w;
};

struct node{
    int d,u;
    bool operator<(const node &a) const{ return d > a.d; }
    node(int d, int u): d(d) , u(u) {}
};

vector<vector<edge>>g;
vector<int>vis, cnt;
vector<ll>d, h;
int n, m;

bool spfa(int s){
    h.assign(n + 1, INF);
    vis.assign(n + 1, 0);
    cnt.assign(n + 1, 0);
    h[s] = 0, vis[s] = 1;
    queue<int>q;
    q.push(s);
    while(!q.empty()){
        int u = q.front();
        q.pop();
        vis[u] = 0;
        for(auto &i : g[u]){
            auto [v, w] = i;
            if(h[v] > h[u] + w){
                h[v] = h[u] + w;
                cnt[v] = cnt[u] + 1;
                if(cnt[v] > n){
                    return false;
                }
                if(!vis[v]){
                    vis[v] = 1;
                    q.push(v);
                }
            }
        }
    }
    return true;
}

void dijkstra(int s){
     d.assign(n + 1, INF);
     vis.assign(n + 1, 0);
     d[s] = 0;
     priority_queue<node>q;
     q.push({0, s});
     while(!q.empty()){
        int u = q.top().u;
        q.pop();
        if(vis[u]){
            continue;
        }
        vis[u] = 1;
        for(auto &i : g[u]){
            int v = i.v , w = i.w;
            if(d[v] > d[u] + w){
                d[v] = d[u] + w;
                q.push({d[v],v});
            }
        }
     }
     return;
}
```
## 二分图 染色法判图
```cpp
int n, m;
int color[N];
vector<int>e[N];

bool dfs(int u, int c){
    color[u] = c;
    for (auto &i : e[u]) {
        if (!color[i]) {
            if (!dfs(i, 3 - c)) return false;
        }
        else if (color[i] == c) return false;
    }
    return true;
}

// 判图
bool flag = true;
for (int i = 1; i <= n; i ++ ){
    if (!color[i]) {
        if (!dfs(i, 1)) {
            flag = false;
            break;
        }
    }
}
if (flag) puts("Yes");
else puts("No");

```
## 二分图 最大匹配
```cpp
#include<bits/stdc++.h>
using namespace std;
const int N = 1e5 + 5;
#define ll long long

vector<int>edge[505];
vector<int>match(505, 0);
vector<bool>st(505, false);

bool find(int x){
    for(auto &i : edge[x]){
        if(!st[i]){
            st[i] = true;
            if(!match[i] || find(match[i])){
                match[i] = x;
                return true;
            }
        }
    }
    return false;
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(0),cout.tie(0);
    int n, m, e;
    cin >> n >> m >> e;
    int a, b;
    while(e--){
        cin >> a >> b;
        edge[a].push_back(b);
    }
    int sum = 0;
    // 匹配时只需要找其中一个集合
    for(int i = 1; i <= n; i++){
        st.assign(505, false);
        if(find(i))sum++;
    }
    cout << sum << "\n";
    return 0;
}
```

## 二分图 最大匹配边权 KM算法
```cpp
#include<bits/stdc++.h>
using namespace std;
#define ll long long
const int INF = 0x3f3f3f3f;
const int N = 2e5 + 10;

vector<int>match(50), lx(50, 0), ly(50, 0);
vector<bool>sx(50), sy(50);

int a[50][50];
int n, MIN;

bool find(int x) {
    sx[x] = true;
    for (int i = 1; i <= n; i++) {
        if (!sy[i]) {
            if (lx[x] + ly[i] == a[x][i]) {
                sy[i] = true;
                if (!match[i] || find(match[i])) {
                    match[i] = x;
                    return true;
                }
            }
            else if (lx[x] + ly[i] > a[x][i]) {
                MIN = min(MIN, lx[x] + ly[i] - a[x][i]);
            }
        }
    }
    return false;
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(0), cout.tie(0);
    cin >> n;
    int x;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            cin >> x;
            a[i][j] = x;
        }
    }
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            lx[i] = max(lx[i], a[i][j]);       
        }
    }
    for (int i = 1; i <= n; i++) {
        while (1) {
            MIN = INF;
            sx.assign(25, false);
            sy.assign(25, false);
            if (find(i)) break;
            for (int j = 1; j <= n; j++) {
                if(!sx[j])continue;
                lx[j] -= MIN;
            }
            for (int j = 1; j <= n; j++) {
                if(!sy[j])continue;
                ly[j] += MIN;
            }
        }
    }
    int sum = 0;
    for (int i = 1; i <= n; i++) {
        sum += a[match[i]][i];
    }
    cout << sum << "\n";
    return 0; 
}
```

# 数论

## 素数筛

```cpp
void get_primes(int n)
{
    int primes[1], cnt;     // primes[]存储所有素数  
    bool st[1];         // st[x]存储x是否被筛掉
    //上面两行开全局,1为N
    for (int i = 2; i <= n; i ++ )
    {
        if (!st[i]) primes[cnt ++ ] = i;
        for (int j = 0; primes[j] <= n / i; j ++ )
        {
            st[primes[j] * i] = true;
            if (i % primes[j] == 0) break;
        }
    }
}
```

## 分解质因数

```cpp
void divide(int x)
{
    for (int i = 2; i <= x / i; i ++ )
        if (x % i == 0)
        {
            int s = 0;
            while (x % i == 0) x /= i, s ++ ;
            cout << i << ' ' << s << endl;
        }
    if (x > 1) cout << x << ' ' << 1 << endl;
    cout << endl;
}
```
## 最大公约数

```cpp
int gcd(int a, int b){
    return b ? gcd(b, a % b) : a;
}
```

## 扩展欧几里得算法

```cpp
// 求x, y，使得ax + by = gcd(a, b)
int exgcd(int a, int b, int &x, int &y) {
    if (!b) {
        x = 1; y = 0;
        return a;
    }
    int d = exgcd(b, a % b, y, x);
    y -= (a/b) * x;
    return d;
}
```
## 快速幂
```cpp
int qmi(int m, int k, int p) {
    int res = 1 % p, t = m;
    while (k) {
        if (k&1) res = res * t % p;
        t = t * t % p;
        k >>= 1;
    }
    return res;
}
```

## 卡特兰数

```cpp
Cat[n] = C(2, n) / (n + 1);
```

# 数据结构

## 邻接表 链式前向星
```cpp
int h[N], w[N], e[N], ne[N], idx;

void add(int a, int b, int c){
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx++ ;
}
```

## 树状数组

```cpp 

int tr1[N];  // 维护a[i]前缀和
int tr2[N];  // 维护a[i] * i的前缀和

int lowbit(int x) {
    return x & (-x);
}

void add(int tr[], int x, int k) {
    for (int i = x; i <= n; i += lowbit(i)) {
        tr[i] += k;
    }
}

// 求某点值
int sum(int tr[], int x) {
    int res = 0;
    for (int i = x; i ; i -= lowbit(i)) {
        sum += tr[i];    
    }
} 

// 求区间和
int prefix_sum(int x){
    return sum(tr1, x) * (x + 1) - sum(tr2, x);
}

//操作1 求单点值
cout << sum(l) << "\n";

// 建树
add(l, d), add(r + 1, -d);

//操作2 求区间和
cout << prefix_sum(r) - prefix_sum(l - 1);

//建树        
add(tr1, l, r), add(tr2, l, l * r);
add(tr1, l + 1, - r), add(tr2, l + 1, (l + 1) * - r);

```
## 线段树

```cpp

//本模板的操作为 区间乘一个x, 区间加一个x

ll tr[N * 4], b[N * 4], mul[N * 4];
ll a[N];
ll mod;

void build(int p, int l, int r){
    mul[p] = 1;
    if(l == r){
        tr[p] = a[l];
        return;
    }
    int mid = l + ((r - l) >> 1);
    build(2 * p, l, mid), build(2 * p + 1, mid + 1, r);
    tr[p] = (tr[p * 2] + tr[p * 2 + 1]) % mod;
    return;
}

void pushup(int p){
    tr[p] = (tr[p * 2] + tr[p * 2 + 1]) % mod;
    return;
}

void pushdowm(int p, int s, int t){
    int m = s + ((t - s) >> 1);
    // 乘法懒惰标记
    if(mul[p] != 1){
        mul[p * 2] = mul[p * 2] * mul[p] % mod;
        mul[p * 2 + 1] = mul[p * 2 + 1] * mul[p] % mod;
        b[p * 2] = b[p * 2] * mul[p] % mod;
        b[p * 2 + 1] = b[p * 2 + 1] * mul[p] % mod;
        tr[p * 2] = tr[p * 2] * mul[p] % mod;
        tr[p * 2 + 1] = tr[p * 2 + 1] * mul[p] % mod;
        mul[p] = 1;
    }
    // 只有加法时候的模板
    if(b[p]){
        tr[p * 2] = (b[p] * (m - s + 1) + tr[p * 2]) % mod;
        tr[p * 2 + 1] = (b[p] * (t - m) + tr[p * 2 + 1]) % mod;
        b[p * 2] = (b[p] + b[p * 2]) % mod;
        b[p * 2 + 1] = (b[p] + b[p * 2 + 1]) % mod;
        b[p] = 0;
    }
    return;
}

void update1(int l, int r, ll c, int s, int t, int p){
    if(l <= s && t <= r){
        tr[p] += (t - s + 1) * c, b[p] += c;
        tr[p] %= mod, b[p] %= mod;
        return;
    }  
    pushdowm(p, s, t);
    int m = s + ((t - s) >> 1);
    if (l <= m) update1(l, r, c, s, m, p * 2);
    if (r > m) update1(l, r, c, m + 1, t, p * 2 + 1);
    pushup(p);
    return;
}

void update2(int l, int r, ll c, int s, int t, int p){
    if(l <= s && t <= r){
        tr[p] *=  c, b[p] *= c, mul[p] *= c;
        tr[p] %= mod, b[p] %= mod, mul[p] %= mod;
        return;
    }  
    pushdowm(p, s, t);
    int m = s + ((t - s) >> 1);
    if (l <= m) update2(l, r, c, s, m, p * 2);
    if (r > m) update2(l, r, c, m + 1, t, p * 2 + 1);
    pushup(p);
    return;
}

ll getsum(int l, int r, int s, int t, int p) {
    if (l <= s && t <= r) return tr[p];
    int m = s + ((t - s) >> 1);
    pushdowm(p, s, t);
    ll sum = 0;
    if (l <= m) sum += getsum(l, r, s, m, p * 2);
    sum %= mod;
    if (r > m) sum += getsum(l, r, m + 1, t, p * 2 + 1);
    return sum % mod;
}

```

## 模拟散列表之开放寻址法

```cpp
ull find(ull x) {
    ull t = (x % N + N) % N;
    while (h[t] != INF && h[t] != x) {
        t++ ;
        if (t == N) t = 0;
    }
    return t;
}
```
## 字符串HASH

```cpp
#define ull unsigned long long
const int N = 1e5 + 11;
const ull P = 131;

ull h[N], p[N], a[N];
string s;

ull get(int l, int r)
{
    return h[r] - h[l - 1] * p[r - l + 1];
}

ull Hash(){
    int k = s.size();
    p[0] = 1;
    for (int i = 1; i <= k; i ++ )
    {
        h[i] = h[i - 1] * P + (ull)s[i - 1];
        p[i] = p[i - 1] * P;
    }
    return get(1, k);
}

//如果爆ll :
#define ull unsigned long long
const ull P = 233;
const int N = 2e6 + 10;
const ull mod = 99989397;

ull h[N], p[N], a[N];
string s;

ull get(int l, int r)
{
    return (h[r] - h[l - 1] * p[r - l + 1] % mod + mod) % mod;
}

ull Hash(){
    int k = s.size();
    p[0] = 1;
    for (int i = 1; i <= k; i ++ )
    {
        h[i] = ((h[i - 1] * P) % mod + (ull)s[i - 1]) % mod;
        p[i] = (p[i - 1] * P) % mod;
    }
    return get(1, k);
}
```

## Trie树

```cpp
int nex[N][65], cnt, a[N];
int n, m;

int getval(char c) {
    if (c >= 'A' && c <= 'Z') {
        return (c - 'A');
    }
    else if (c >= 'a' && c <= 'z') {
        return (c - 'a' + 26);
    }
    else {
        return (c - '0' + 52);
    }
}

void insert(string s) {
    int p = 0;
    for (int i = 0; i < s.size(); i++) {
        int c = getval(s[i]);
        if (!nex[p][c]) nex[p][c] = ++cnt;
        p = nex[p][c];
        a[p]++;
    }
    return;
}

int find(string s) {
    int p = 0;
    for (int i = 0; i < s.size(); i++) {
        int c = getval(s[i]);
        if (!nex[p][c]) return 0;
        p = nex[p][c];
    }
    return a[p];
}
```

## Trie数之最大异或与最小异或

```cpp
#define ll long long
const int N = 1e5 + 10;
#define PII pair<int, int>
const ll INF = 0x3f3f3f3f;

int nex[N * 32][2], a[N * 32], ep[N * 32];
int cnt;

void insert(ll s) {
	int p = 0;
	for (int i = 31;  i >= 0 ; i--) {
		int c = s >> i & 1;
		if (nex[p][c] == -1)
			nex[p][c] = ++cnt;
		p = nex[p][c];
	}
	return;
}

//最大异或
ll find1(ll s) {
	ll ans = s;
	int p = 0;
	for (int i = 31; i >= 0; i--) {
		int c = s >> i & 1;
		if (nex[p][!c] != -1) {
			ans ^= ((!c) << i);
			p = nex[p][!c];
		} else {
			ans ^= (c << i);
			p = nex[p][c];
		}
	}
	return ans;
}

// 最小异或
ll find2(ll s) {
	ll ans = s;
	int p = 0;
	for (int i = 31; i >= 0; i--) {
		int c = s >> i & 1;
		if (nex[p][c] != -1) {
			ans ^= ((c) << i);
			p = nex[p][c];
		} else {
			ans ^= ((!c) << i);
			p = nex[p][!c];
		}
	}
	return ans;
}

void solve(int u) {
	memset(nex, -1, sizeof nex);
	cnt = 0;
	int n;
	cin >> n;
	ll sum = 0;
	ll ans1 = -INF;
	ll ans2 = INF;
	insert(0);
	ll x;
	while (n--) {
		cin >> x;
		sum ^= x;
		ans2 = min(ans2, find2(sum));
		insert(sum);
		ans1 = max(ans1, find1(sum));
	}
	cout << "Case " << u << ": " << ans1 << " " << ans2 << "\n";
	return;
}

```
